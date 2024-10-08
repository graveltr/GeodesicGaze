//
//  MultiCamCapture.swift
//  MultiCamDemo
//
//  Created by Trevor Gravely on 7/15/24.
//

import AVFoundation
import UIKit
import Foundation
import os

protocol MultiCamCaptureDelegate: AnyObject {
    func processCameraPixelBuffers(frontCameraPixelBuffer: CVPixelBuffer, backCameraPixelBuffer: CVPixelBuffer)
}

class MultiCamCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var multiCamSession: AVCaptureMultiCamSession!
    private var frontCameraPreviewLayer: AVCaptureVideoPreviewLayer!
    private var backCameraPreviewLayer: AVCaptureVideoPreviewLayer!
    
    private var frontCameraConnection: AVCaptureConnection?
    private var backCameraConnection: AVCaptureConnection?
    
    weak var delegate: MultiCamCaptureDelegate?
    
    let logger = Logger()

    override init() {
        super.init()
        setupMultiCamSession()
    }
    
    private func setupMultiCamSession() {
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            fatalError("Multi cam not supported")
        }
        multiCamSession = AVCaptureMultiCamSession()
        
        // Configure the front camera.
        guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let frontCameraInput = try? AVCaptureDeviceInput(device: frontCamera),
              multiCamSession.canAddInput(frontCameraInput) else {
            fatalError("Unable to add front camera input")
        }
        multiCamSession.addInputWithNoConnections(frontCameraInput)
        
        let frontCameraOutput = AVCaptureVideoDataOutput()
        frontCameraOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "frontCameraQueue"))
        if multiCamSession.canAddOutput(frontCameraOutput) {
            print("Can add front camera output")
        }
        multiCamSession.addOutputWithNoConnections(frontCameraOutput)
        
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let backCameraInput = try? AVCaptureDeviceInput(device: backCamera),
              multiCamSession.canAddInput(backCameraInput) else {
            fatalError("Unable to add back camera input")
        }
        multiCamSession.addInputWithNoConnections(backCameraInput)
        
        let backCameraOutput = AVCaptureVideoDataOutput()
        backCameraOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "backCameraQueue"))
        if multiCamSession.canAddOutput(backCameraOutput) {
            print("Can add back camera output")
        }
        multiCamSession.addOutputWithNoConnections(backCameraOutput)
        
        frontCameraConnection = AVCaptureConnection(inputPorts: frontCameraInput.ports, output: frontCameraOutput)
        if multiCamSession.canAddConnection(frontCameraConnection!) {
            multiCamSession.addConnection(frontCameraConnection!)
        } else {
            fatalError("Unable to add front camera connection")
        }
        
        backCameraConnection = AVCaptureConnection(inputPorts: backCameraInput.ports, output: backCameraOutput)
        if multiCamSession.canAddConnection(backCameraConnection!) {
            multiCamSession.addConnection(backCameraConnection!)
        } else {
            fatalError("Unable to add back camera connection")
        }
        
        // Setup preview layers
        frontCameraPreviewLayer = AVCaptureVideoPreviewLayer(sessionWithNoConnection: multiCamSession)
        backCameraPreviewLayer = AVCaptureVideoPreviewLayer(sessionWithNoConnection: multiCamSession)
        
        // Add connections to preview layers
        if let frontPort = frontCameraInput.ports.first {
            let frontPreviewConnection = AVCaptureConnection(inputPort: frontPort, videoPreviewLayer: frontCameraPreviewLayer)
            if multiCamSession.canAddConnection(frontPreviewConnection) {
                multiCamSession.addConnection(frontPreviewConnection)
            }
        }
        
        if let backPort = backCameraInput.ports.first {
            let backPreviewConnection = AVCaptureConnection(inputPort: backPort, videoPreviewLayer: backCameraPreviewLayer)
            if multiCamSession.canAddConnection(backPreviewConnection) {
                multiCamSession.addConnection(backPreviewConnection)
            }
        }
    }
    
    func startRunning() {
        logger.info("Starting run ...")
        multiCamSession.startRunning()
    }
    
    func stopRunning() {
        multiCamSession.stopRunning()
    }
    
    func setupPreviewLayers(frontView: UIView, backView: UIView) {
        logger.info("Setting up preview layers ...")
        frontCameraPreviewLayer.videoGravity = .resizeAspect
        frontCameraPreviewLayer.frame = frontView.bounds
        frontView.layer.addSublayer(frontCameraPreviewLayer)
        
        backCameraPreviewLayer.videoGravity = .resizeAspect
        backCameraPreviewLayer.frame = backView.bounds
        backView.layer.addSublayer(backCameraPreviewLayer)
    }
    
    // Will need to cache one of the sample buffers so mixer can be
    // called with both of the buffers.
    private var currentBackCameraSampleBuffer: CMSampleBuffer?

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        logger.debug("Running capture output ...")
        if connection == frontCameraConnection {
            processFrontCameraBuffer(sampleBuffer)
        } else if connection == backCameraConnection {
            processBackCameraBuffer(sampleBuffer)
        }
    }
    
    private func processBackCameraBuffer(_ backCameraSampleBuffer: CMSampleBuffer) {
        logger.info("Caching sample buffer from back camera")
        currentBackCameraSampleBuffer = backCameraSampleBuffer
    }
    
    private func processFrontCameraBuffer(_ frontCameraSampleBuffer: CMSampleBuffer) {
        logger.info("Processing pixel buffer from front camera")
        
        guard let frontCameraPixelBuffer = CMSampleBufferGetImageBuffer(frontCameraSampleBuffer) else {
            fatalError("Failed to retrieve pixel buffer from front camera sample buffer")
        }
        
        guard let backCameraSampleBuffer = currentBackCameraSampleBuffer else {
            logger.info("Cached back camera sample buffer is nil, returning ...")
            return
        }
        guard let backCameraPixelBuffer = CMSampleBufferGetImageBuffer(backCameraSampleBuffer) else {
            fatalError("Failed to retrieve pixel buffer from current back camera sample buffer")
        }
        
        delegate?.processCameraPixelBuffers(frontCameraPixelBuffer: frontCameraPixelBuffer,
                                            backCameraPixelBuffer: backCameraPixelBuffer)
    }
}
