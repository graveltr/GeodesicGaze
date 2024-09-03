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

struct SchwarzschildLenseResult {
    var varphitilde: Float
    var ccw: Bool
    var status: Int32
}

protocol BufferData {
    func createBuffer(device: MTLDevice) -> MTLBuffer?
}

extension Array: BufferData where Element == Float {
    func createBuffer(device: MTLDevice) -> MTLBuffer? {
        return device.makeBuffer(bytes: self, length: MemoryLayout<Float>.size * self.count, options: []);
    }
}

struct AnyBufferData: BufferData {
    private let _createBuffer: (MTLDevice) -> MTLBuffer?
    
    init<T: BufferData>(_ bufferData: T) {
        _createBuffer = bufferData.createBuffer
    }
    
    func createBuffer(device: MTLDevice) -> MTLBuffer? {
        return _createBuffer(device)
    }
}

// TODO: more robust testing of elliptic F -> what happens outside the bounds?
final class GeodesicGazeTests: XCTestCase {

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var computePipelineState: MTLComputePipelineState!
    var library: MTLLibrary!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        library = device.makeDefaultLibrary()
        
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
        let testAngles: [Float] = [
            0.0,            // Lower boundary
            0.1,            // Small angle
            0.5,            // Mid-range angle
            1.0,            // Larger angle
            1.5,            // Near upper boundary (π/2)
            Float.pi / 4   // π/4 (45 degrees)
        ]

        let testModuli: [Float] = [
            0.0,    // Lower boundary (no elliptic effect)
            0.1,    // Small modulus
            0.3,    // Small to mid-range modulus
            0.5,    // Mid-range modulus
            0.7,    // Mid to high-range modulus
            0.9    // High-range modulus
        ]
        
        let wrappedTestAngles = AnyBufferData(testAngles)
        let wrappedTestModuli = AnyBufferData(testModuli)
        let inputs: [AnyBufferData] = [wrappedTestAngles, wrappedTestModuli]
        
        let count = testAngles.count
        let resultBufferSize = count * MemoryLayout<EllintResult>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "ellint_F_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: EllintResult.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        let expectedResults = readExpectedResults()
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult.val, expectedResult.val, accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult.status, expectedResult.status, "Non SUCCESSful status code at index \(i)")
        }
    }
    
    func runComputeShader(shaderName: String, inputs: [AnyBufferData], resultsBuffer: MTLBuffer) {
        let kernelFunction = library?.makeFunction(name: shaderName)
        computePipelineState = try! device.makeComputePipelineState(function: kernelFunction!)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        computeEncoder.setComputePipelineState(computePipelineState)
        
        var buffers: [MTLBuffer?] = []
        for (index, input) in inputs.enumerated() {
            if let buffer = input.createBuffer(device: device) {
                computeEncoder.setBuffer(buffer, offset: 0, index: index)
                buffers.append(buffer)
            } else {
                XCTFail("Failed to create buffer for input group \(index).")
            }
        }
        
        computeEncoder.setBuffer(resultsBuffer, offset: 0, index: inputs.count)

        // Set the number of threads
        let threadsPerThreadgroup = MTLSize(width: 8, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (inputs.count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                   height: 1,
                                   depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
