import SwiftUI
import AVFoundation
import MetalKit

struct MultiCamView: UIViewControllerRepresentable {
    class Coordinator: NSObject, MultiCamCaptureDelegate, MTKViewDelegate {
        var parent: MultiCamView
        var mixer: BhiMixer
        var mtkView: MTKView
        var multiCamCapture: MultiCamCapture

        init(parent: MultiCamView, mtkView: MTKView, mixer: BhiMixer, multiCamCapture: MultiCamCapture) {
            self.parent = parent
            self.mtkView = mtkView
            self.mixer = mixer
            self.multiCamCapture = multiCamCapture
            super.init()
            self.mtkView.delegate = self
        }
        
        func processCameraPixelBuffers(frontCameraPixelBuffer: CVPixelBuffer, backCameraPixelBuffer: CVPixelBuffer) {
            mixer.mix(frontCameraPixelBuffer: frontCameraPixelBuffer,
                      backCameraPixelBuffer: backCameraPixelBuffer,
                      in: mtkView)
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
          mixer.createLutTexture(width: Int(size.width), height: Int(size.height))
          mixer.precomputeLutTexture()
        }
        
        func draw(in view: MTKView) {}
    }

    func makeCoordinator() -> Coordinator {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.framebufferOnly = false
        let mixer = BhiMixer(device: mtkView.device!)
        let multiCamCapture = MultiCamCapture()
        
        return Coordinator(parent: self, mtkView: mtkView, mixer: mixer, multiCamCapture: multiCamCapture)
    }

    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        
        // let videoViewHeight = viewController.view.bounds.height / 3
        
        // Configure the views.
        /*
        let frontView = UIView(frame: CGRect(x: 0, y: 0, width: viewController.view.bounds.width, height: videoViewHeight))
        let backView = UIView(frame: CGRect(x: 0, y: videoViewHeight, width: viewController.view.bounds.width, height: videoViewHeight))
        let mtkView = context.coordinator.mtkView
        mtkView.frame = CGRect(x: 0, y: 2 * videoViewHeight, width: viewController.view.bounds.width, height: videoViewHeight)
        */
        
        /*
        let lensedViewHeight = viewController.view.bounds.height * 0.8;
        let previewViewHeight = viewController.view.bounds.height * 0.2;
        
        let frontView = UIView(frame: CGRect(x: viewController.view.bounds.width / 2,
                                             y: 0,
                                             width: viewController.view.bounds.width,
                                             height: previewViewHeight))
        let backView = UIView(frame: CGRect(x: 0,
                                            y: 0,
                                            width: viewController.view.bounds.width / 2,
                                            height: previewViewHeight))
        
        let mtkView = context.coordinator.mtkView
        mtkView.frame = CGRect(x: 0, 
                               y: previewViewHeight,
                               width: viewController.view.bounds.width,
                               height: lensedViewHeight)
        
        viewController.view.addSubview(frontView)
        viewController.view.addSubview(backView)
        viewController.view.addSubview(mtkView)

        let multiCamCapture = context.coordinator.multiCamCapture
        multiCamCapture.setupPreviewLayers(frontView: frontView, backView: backView)
        multiCamCapture.delegate = context.coordinator
        multiCamCapture.startRunning()
         */
        
        let mtkView = context.coordinator.mtkView
        mtkView.frame = CGRect(x: 0,
                               y: 0,
                               width:   viewController.view.bounds.width,
                               height:  viewController.view.bounds.height)
        
        viewController.view.addSubview(mtkView)

        let multiCamCapture = context.coordinator.multiCamCapture
        multiCamCapture.delegate = context.coordinator
        multiCamCapture.startRunning()

        return viewController
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Update the UI view controller if needed
    }
}
