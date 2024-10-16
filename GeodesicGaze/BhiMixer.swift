//
//  BhiMixer.swift
//  MultiCamDemo
//
//  Created by Trevor Gravely on 7/16/24.
//

import MetalKit

class BhiMixer {
    
    struct Uniforms {
        var frontTextureWidth: Int32
        var frontTextureHeight: Int32
        var backTextureWidth: Int32
        var backTextureHeight: Int32
        var mode: Int32
    }
    
    struct PreComputeUniforms {
        var mode: Int32
    }
    
    struct FilterParameters {
        var spaceTimeMode: Int32
        var sourceMode: Int32
        var d: Float
        var a: Float
        var thetas: Float
    }

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var precomputeCommandQueue: MTLCommandQueue!
    var textureCache: CVMetalTextureCache!
    var pipelineState: MTLRenderPipelineState!
    var computePipelineState: MTLComputePipelineState!
    var frontYTexture: MTLTexture?
    var frontUVTexture: MTLTexture?
    var frontTextureHeight: Int?
    var frontTextureWidth: Int?
    var backYTexture: MTLTexture?
    var backUVTexture: MTLTexture?
    var backTextureHeight: Int?
    var backTextureWidth: Int?
    var mode: Int32!
    
    var lutTexture: MTLTexture!
    var lutTextureTemp: MTLTexture!
    
    var selectedFilter = 0 {
        didSet {
            needsNewLutTexture = true
        }
    }
    
    var filterParameters = FilterParameters(spaceTimeMode: 1, sourceMode: 1, d: 0, a: 0, thetas: 0)
    var needsNewLutTexture = true

    // private let accessQueue = DispatchQueue(label: "com.myapp.lutTextureAccessQueue")
    private var semaphore = DispatchSemaphore(value: 1)

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        self.precomputeCommandQueue = device.makeCommandQueue()
        self.mode = 0
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        setupPipelines()
    }
    
    private func setupPipelines() {
        let library = device.makeDefaultLibrary()
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "preComputedFragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Pipeline state created successfully")
        } catch {
            fatalError("Failed to create pipeline state")
        }
        
        let computeFunction = library?.makeFunction(name: "precomputeLut")
        computePipelineState = try! device.makeComputePipelineState(function: computeFunction!)
    }
    
    func createLutTexture(width: Int, height: Int) {
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = width
        descriptor.height = height
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        lutTexture = device.makeTexture(descriptor: descriptor)
        lutTextureTemp = device.makeTexture(descriptor: descriptor)
    }
    
    func precomputeLutTexture(selectedFilter: Int) {
        semaphore.wait()
        let commandBuffer = precomputeCommandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        computeEncoder.setComputePipelineState(computePipelineState)
        
        var uniforms = PreComputeUniforms(mode: mode)
        let uniformsBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<PreComputeUniforms>.size, options: .storageModeShared)
        
        var matrixWidth = lutTextureTemp.width;
        let matrixHeight = lutTextureTemp.height;
        
        print("width: \(matrixWidth), height: \(matrixHeight)")
        
        let totalElements = matrixWidth * matrixHeight;
        
        let bufferSize = totalElements * MemoryLayout<simd_float3>.stride;
        guard let matrixBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            fatalError("Couldn't create buffer")
        }
        
        let widthBuffer = device.makeBuffer(bytes: &matrixWidth, length: MemoryLayout<UInt>.size, options: .storageModeShared)
        
        computeEncoder.setBuffer(uniformsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(matrixBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(widthBuffer, offset: 0, index: 2)
        
        // Write to the temporary texture
        computeEncoder.setTexture(lutTextureTemp, index: 0)
        
        /*
         * lutTexture.dimension + groupSize - 1 / groupSize is just
         * lutTexture.dimension / groupSize rounded up. This guarantees
         * that group * groupSize covers the full texture width / height.
         */
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(width: (lutTextureTemp.width + 15) / 16,
                                   height: (lutTextureTemp.height + 15) / 16,
                                   depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        
        /*
        commandBuffer.addCompletedHandler { _ in
            print("Lut texture computation completed and updated")
        }
        */
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Now copy results from lutTextureTemp to lutTexture using a blit command encoder
        let blitCommandBuffer = precomputeCommandQueue.makeCommandBuffer()!
        guard let blitEncoder = blitCommandBuffer.makeBlitCommandEncoder() else {
            fatalError("Failed to create blit encoder")
        }
        
         
        blitEncoder.copy(from: lutTextureTemp, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                         sourceSize: MTLSize(width: lutTextureTemp.width, 
                                             height: lutTextureTemp.height, 
                                             depth: 1),
                         to: lutTexture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        
        blitEncoder.endEncoding()
        blitCommandBuffer.commit()
        blitCommandBuffer.waitUntilCompleted()
        semaphore.signal()
        
        // semaphore.wait()
        // lutTexture = lutTextureTemp
        // semaphore.signal()

        /*
        commandBuffer.waitUntilCompleted()
        
        let dataPointer = matrixBuffer.contents().bindMemory(to: simd_float3.self, capacity: totalElements)
        let matrixData = Array(UnsafeBufferPointer(start: dataPointer, count: totalElements))
        
        var matrixResult: [[simd_float3]] = Array(repeating: Array(repeating: simd_float3(0,0,0), count: matrixWidth), count: matrixHeight);
        
        for y in 0..<matrixHeight {
            for x in 0..<matrixWidth {
                matrixResult[y][x] = matrixData[y * matrixWidth + x]
            }
        }
        
        for (i, row) in matrixResult.enumerated() {
            var numError = 0;
            for (_, vector) in row.enumerated() {
                if (vector.z == -1) {
                    numError += 1;
                }
            }
            print("row: \(i) numError: \(numError)")
        }
        
        _ = matrixResult[0][0]
        */
    }
    
    func mix(frontCameraPixelBuffer: CVPixelBuffer?,
             backCameraPixelBuffer: CVPixelBuffer?,
             in view: MTKView) {
        
        guard let drawable = view.currentDrawable else {
            print("Currentdrawable is nil")
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Couldn't create command buffer")
            return
        }
        
        if needsNewLutTexture {
            print("computing new lut texture")
            
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            computeEncoder.setComputePipelineState(computePipelineState)
            
            var otherFilterParameters = FilterParameters(spaceTimeMode: filterParameters.spaceTimeMode,
                                                         sourceMode: filterParameters.sourceMode,
                                                         d: filterParameters.d,
                                                         a: filterParameters.a,
                                                         thetas: filterParameters.thetas)
            let filterParametersBuffer = device.makeBuffer(bytes: &otherFilterParameters,
                                                           length: MemoryLayout<FilterParameters>.size,
                                                           options: .storageModeShared)

            var matrixWidth = lutTexture.width;
            let matrixHeight = lutTexture.height;
            let totalElements = matrixWidth * matrixHeight;
            let bufferSize = totalElements * MemoryLayout<simd_float3>.stride;
            
            let matrixBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
            let widthBuffer = device.makeBuffer(bytes: &matrixWidth, length: MemoryLayout<UInt>.size, options: .storageModeShared)
            
            computeEncoder.setBuffer(filterParametersBuffer,    offset: 0, index: 0)
            computeEncoder.setBuffer(matrixBuffer,              offset: 0, index: 1)
            computeEncoder.setBuffer(widthBuffer,               offset: 0, index: 2)
            computeEncoder.setTexture(lutTexture,                          index: 0)
            
            /*
             * lutTexture.dimension + groupSize - 1 / groupSize is just
             * lutTexture.dimension / groupSize rounded up. This guarantees
             * that group * groupSize covers the full texture width / height.
             */
            let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let threadGroups = MTLSize(width: (lutTexture.width + 15) / 16,
                                       height: (lutTexture.height + 15) / 16,
                                       depth: 1)
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            computeEncoder.endEncoding()
            
            needsNewLutTexture = false
        }
        
        if let frontCameraPixelBuffer = frontCameraPixelBuffer {
            let textures = createTexture(from: frontCameraPixelBuffer)
            frontYTexture = textures?.0
            frontUVTexture = textures?.1
            frontTextureWidth = textures?.2
            frontTextureHeight = textures?.3
        }
        if let backCameraPixelBuffer = backCameraPixelBuffer {
            let textures = createTexture(from: backCameraPixelBuffer)
            backYTexture = textures?.0
            backUVTexture = textures?.1
            backTextureWidth = textures?.2
            backTextureHeight = textures?.3
        }
        
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let frontYTexture = frontYTexture,
              let frontUVTexture = frontUVTexture,
              let frontTextureWidth = frontTextureWidth,
              let frontTextureHeight = frontTextureHeight,
              let backYTexture = backYTexture,
              let backUVTexture = backUVTexture,
              let backTextureWidth = backTextureWidth,
              let backTextureHeight = backTextureHeight else {
            print("returning from mix")
            return
        }
        
        var uniforms = Uniforms(frontTextureWidth: Int32(frontTextureWidth), 
                                frontTextureHeight: Int32(frontTextureHeight),
                                backTextureWidth: Int32(backTextureWidth),
                                backTextureHeight: Int32(backTextureHeight),
                                mode: mode)
        let uniformsBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.size, options: .storageModeShared)
        
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        if renderPassDescriptor.colorAttachments[0].texture == nil {
            print("texture is null")
            return
        }

        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setFragmentBuffer(uniformsBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(frontYTexture, index: 0)
        renderEncoder.setFragmentTexture(frontUVTexture, index: 1)
        renderEncoder.setFragmentTexture(backYTexture, index: 2)
        renderEncoder.setFragmentTexture(backUVTexture, index: 3)
        renderEncoder.setFragmentTexture(lutTexture, index: 4)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func createTexture(from pixelBuffer: CVPixelBuffer) -> (MTLTexture?, 
                                                                    MTLTexture?,
                                                                    Int,
                                                                    Int)? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var yTexture: CVMetalTexture?
        var uvTexture: CVMetalTexture?
        
        let yStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .r8Unorm, width, height, 0, &yTexture)
        
        let uvStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .rg8Unorm, width / 2, height / 2, 1, &uvTexture)
        
        if yStatus == kCVReturnSuccess, uvStatus == kCVReturnSuccess {
            let yMetalTexture = CVMetalTextureGetTexture(yTexture!)
            let uvMetalTexture = CVMetalTextureGetTexture(uvTexture!)
            return (yMetalTexture, uvMetalTexture, width, height)
        }

        print("Couldn't create texture")
        return nil
    }
}
