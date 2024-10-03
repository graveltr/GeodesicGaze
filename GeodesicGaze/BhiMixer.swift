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
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
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

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
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
    }
    
    func precomputeLutTexture() {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        computeEncoder.setComputePipelineState(computePipelineState)
        
        var uniforms = PreComputeUniforms(mode: mode)
        let uniformsBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<PreComputeUniforms>.size, options: .storageModeShared)
        
        computeEncoder.setBuffer(uniformsBuffer, offset: 0, index: 0)
        computeEncoder.setTexture(lutTexture, index: 0)
        
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
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func mix(frontCameraPixelBuffer: CVPixelBuffer?,
             backCameraPixelBuffer: CVPixelBuffer?,
             in view: MTKView) {
        print("Mixing ...")
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
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let frontYTexture = frontYTexture,
              let frontUVTexture = frontUVTexture,
              let frontTextureWidth = frontTextureWidth,
              let frontTextureHeight = frontTextureHeight,
              let backYTexture = backYTexture,
              let backUVTexture = backUVTexture,
              let backTextureWidth = backTextureWidth,
              let backTextureHeight = backTextureHeight,
              let drawable = view.currentDrawable else {
            print("returning from mix")
            return
        }
        print("Successfully created command buffer and render pass descriptor")
        
        var uniforms = Uniforms(frontTextureWidth: Int32(frontTextureWidth), 
                                frontTextureHeight: Int32(frontTextureHeight),
                                backTextureWidth: Int32(backTextureWidth),
                                backTextureHeight: Int32(backTextureHeight),
                                mode: mode)
        
        print("backWidth: \(uniforms.backTextureWidth) backHeight: \(uniforms.backTextureHeight)")
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

        print("Successfully set fragment textures and buffers")

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
