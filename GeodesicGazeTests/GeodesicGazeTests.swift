//
//  GeodesicGazeTests.swift
//  GeodesicGazeTests
//
//  Created by Trevor Gravely on 9/1/24.
//

@testable import GeodesicGaze
import XCTest
import Metal
import simd

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

struct PhiSResult {
    var val: Float
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
    
    func testEllintFInBounds() {
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
    
    func testEllintFOutOfBounds() {
        let testAngles: [Float] = [Float.pi / 2.0]
        let testModuli: [Float] = [sqrt(2)]
        
        let wrappedTestAngles = AnyBufferData(testAngles)
        let wrappedTestModuli = AnyBufferData(testModuli)
        let inputs: [AnyBufferData] = [wrappedTestAngles, wrappedTestModuli]
        
        let count = testAngles.count
        let resultBufferSize = count * MemoryLayout<EllintResult>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "ellint_F_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: EllintResult.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        let result1 = EllintResult(val: 0.0, err: 0.0, status: -1)
        let expectedResults = [result1]
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult.status, expectedResult.status, "Unexpected status code at index \(i)")
        }
    }
    
    func testRadialRoots() {
        let M: [Float] = [1.0, 1.0, 1.0]
        let b: [Float] = [5.2, 5.31, 10.11]
        
        let wrappedM = AnyBufferData(M)
        let wrappedb = AnyBufferData(b)
        let inputs: [AnyBufferData] = [wrappedM, wrappedb]
        
        let count = M.count
        let resultBufferSize = count * MemoryLayout<simd_float4>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "radial_roots_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: simd_float4.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        // Generated via Mathematica
        let expectedResults = [simd_float4(-6.00395, 0, 2.93529, 3.06866),
                               simd_float4(-6.11681, 0, 2.69151, 3.42529),
                               simd_float4(-10.9914, 0, 2.08922, 8.90217)]
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult[0], expectedResult[0], accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult[1], expectedResult[1], accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult[2], expectedResult[2], accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult[3], expectedResult[3], accuracy: 1e-3, "Value mismatch at index \(i)")
        }
    }
    
    func testPhiS() {
        let M:  [Float] = [1.0, 1.0, 1.0, 1.0]
        let ro: [Float] = [30.0, 30.0, 30.0, 30.0]
        let rs: [Float] = [1000.0, 1000.0, 1000.0, 1000.0]
        let b:  [Float] = [5.2, 5.31, 10.11, 21.2132034355]
        
        let wrappedM = AnyBufferData(M)
        let wrappedro = AnyBufferData(ro)
        let wrappedrs = AnyBufferData(rs)
        let wrappedb = AnyBufferData(b)
        let inputs: [AnyBufferData] = [wrappedM, wrappedro, wrappedrs, wrappedb]
        
        let count = M.count
        let resultBufferSize = count * MemoryLayout<PhiSResult>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "phiS_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: PhiSResult.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        let expectedResults = [
            PhiSResult(val: 9.7726,     status: 0),
            PhiSResult(val: 6.42818,    status: 0),
            PhiSResult(val: 3.3687,     status: 0),
            PhiSResult(val: 2.56081019, status: 0)
        ]
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult.val, expectedResult.val, accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult.status, expectedResult.status, "Non SUCCESSful status code at index \(i)")
        }
    }
    
    func testNormalizeAngle() {
        let phi:      [Float] = [2.56081019]
        
        let wrappedphi = AnyBufferData(phi)
        let inputs: [AnyBufferData] = [wrappedphi]
        
        let count = phi.count
        let resultBufferSize = count * MemoryLayout<Float>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "normalize_angle_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: Float.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        let expectedResults: [Float] = [2.56081019]
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult, expectedResult, accuracy: 1e-3, "Value mismatch at index \(i)")
        }
    }
    
    func testSchwarzchildLense() {
        let M:      [Float] = [1.0]
        let ro:     [Float] = [30.0]
        let rs:     [Float] = [1000.0]
        let varphi: [Float] = [Float.pi / 4.0]
        let bs: [Float] = [30.0 * sin(Float.pi / 4.0)]
        
        let wrappedM = AnyBufferData(M)
        let wrappedro = AnyBufferData(ro)
        let wrappedrs = AnyBufferData(rs)
        let wrappedbs = AnyBufferData(bs)
        let inputs: [AnyBufferData] = [wrappedM, wrappedro, wrappedrs, wrappedbs]
        
        let count = M.count
        let resultBufferSize = count * MemoryLayout<SchwarzschildLenseResult>.size
        let resultsBuffer = device.makeBuffer(length: resultBufferSize, options: [])

        runComputeShader(shaderName: "schwarzschild_lense_compute_kernel", inputs: inputs, resultsBuffer: resultsBuffer!)
        
        let resultsPointer = resultsBuffer?.contents().bindMemory(to: SchwarzschildLenseResult.self, capacity: count)
        let gpuResults = Array(UnsafeBufferPointer(start: resultsPointer, count: count))
        
        let expectedResults = [
            SchwarzschildLenseResult(varphitilde: 0.56472623, ccw: false, status: 0)
        ]
        
        for i in 0..<gpuResults.count {
            let gpuResult = gpuResults[i]
            let expectedResult = expectedResults[i]
            XCTAssertEqual(gpuResult.varphitilde, expectedResult.varphitilde, accuracy: 1e-3, "Value mismatch at index \(i)")
            XCTAssertEqual(gpuResult.ccw, expectedResult.ccw, "CCW mismatch at index \(i)")
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
