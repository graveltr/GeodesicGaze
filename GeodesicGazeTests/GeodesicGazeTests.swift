//
//  GeodesicGazeTests.swift
//  GeodesicGazeTests
//
//  Created by Trevor Gravely on 9/1/24.
//

@testable import GeodesicGaze
import XCTest
import Metal

struct EllintResult {
    var val: Float
    var err: Float 
    var status: Int32
}

final class GeodesicGazeTests: XCTestCase {

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var computePipelineState: MTLComputePipelineState!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()

        // Load the Metal library containing your compute shader
        let library = device.makeDefaultLibrary()
        let kernelFunction = library?.makeFunction(name: "ellint_F_compute_kernel")
        computePipelineState = try! device.makeComputePipelineState(function: kernelFunction!)
    }
    
    func readExpectedResults() -> [EllintResult] {
        guard let fileURL = Bundle(for: type(of: self)).url(forResource: "expected_results", withExtension: "txt") else {
            fatalError("Expected results file not found")
        }
        
        guard let content = try? String(contentsOf: fileURL) else {
            fatalError("Unable to read expected results file")
        }
        
        var expectedResults: [EllintResult] = []
        let lines = content.split(separator: "\n")
        for line in lines {
            let components = line.split(separator: " ").map { Float($0)! }
            let result = EllintResult(val: components[0], err: components[1], status: Int32(components[2]))
            expectedResults.append(result)
        }
        
        return expectedResults
    }
    
    func testEllintF() {
        let testAngles: [Float] = [0.0, 0.5, 1.0]
        let testModuli: [Float] = [0.0, 0.5, 1.0]
        
        var expectedResults = readExpectedResults()
        
        let gpuResults = runEllintFComputeShader(angles: testAngles, moduli: testModuli)
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult.val, expectedResult.val, accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult.status, expectedResult.status, "Non SUCCESSful status code at index \(i)")
        }
    }
    
    func runEllintFComputeShader(angles: [Float], moduli: [Float]) -> [EllintResult] {
        let count = angles.count
        let bufferSize = count * MemoryLayout<Float>.size
        let resultBufferSize = count * MemoryLayout<EllintResult>.size

        // Create buffers
        let anglesBuffer = device.makeBuffer(bytes: angles, length: bufferSize, options: [])
        let moduliBuffer = device.makeBuffer(bytes: moduli, length: bufferSize, options: [])
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        // Create a command buffer and compute command encoder
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(anglesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(moduliBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultsBuffer, offset: 0, index: 2)

        // Set the number of threads
        let threadsPerThreadgroup = MTLSize(width: 8, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, 
                                   height: 1,
                                   depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: EllintResult.self, capacity: count)
        let results = Array(UnsafeBufferPointer(start: resultsPointer, count: count))

        return results
    }
}
