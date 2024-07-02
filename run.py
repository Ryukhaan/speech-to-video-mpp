from Lipsreenact import LipsReenact as Reenacter
import argparse
import os
import gradio as gr

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using HyperLipsBase or HyperLipsHR models')
parser.add_argument('--checkpoint_path_BASE', type=str,help='Name of saved HyperLipsBase checkpoint to load weights from', default="checkpoints/hyperlipsbase_lrs2.pth")
parser.add_argument('--checkpoint_path_HR', type=str,help='Name of saved HyperLipsHR checkpoint to load weights from', default=None)#"checkpoints/hyperlipshr_mead_128.pth"
parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', default="test/video5/video5.mp4")
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', default="test/video5/video5.wav")
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='result/result_video.mp4')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--filter_window', default=None, type=int,
                    help='real window is 2*T+1')
parser.add_argument('--hyper_batch_size', type=int, help='Batch size for hyperlips model(s)', default=128)
parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--segmentation_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/face_segmentation.pth")
parser.add_argument('--face_enhancement_path', type=str,
					help='Name of saved checkpoint of segmentation network', default="checkpoints/GFPGANv1.3.pth")#"checkpoints/GFPGANv1.3.pth"
parser.add_argument('--no_faceenhance', default=False, action='store_true',
					help='Prevent using face enhancement')
parser.add_argument('--gpu_id', type=float, help='gpu id (default: 0)',
                    default=0, required=False)
args = parser.parse_args()

def gui_inference_single(face, audio, hyper_batch_size):
    print(face, type(face))
    executor = Reenacter(checkpoint_path_BASE=args.checkpoint_path_BASE,
                                   checkpoint_path_HR=args.checkpoint_path_HR,
                                   segmentation_path=args.segmentation_path,
                                   face_enhancement_path=args.face_enhancement_path,
                                   gpu_id=args.gpu_id,
                                   window=args.filter_window,
                                   hyper_batch_size=hyper_batch_size,
                                   img_size=args.img_size,
                                   resize_factor=args.resize_factor,
                                   pad=args.pads)
    executor._LoadModels()
    executor._Inference(face, audio, args.outfile)
    return args.outfile, args.outfile

face_enhancement_checkbox = gr.Checkbox(label="Face Enhancement",
            value=True,
            info="Disable face enhancement provides a faster but low quality output.")

text_to_speech_checkbox = gr.Checkbox(label="Text to Speech",
                                      value=False,
                                      info="Only Text is provided. It will automatically generates audio from the text.")

hyper_batch_size_slider = gr.Slider(label="Batch Size",
                                    value=32,
                                    minimum=4,
                                    maximum=128,
                                    step=4)

input_video = gr.Video(label="Upload a video")

audio_input = gr.Audio(label="Upload an audio")

output_video = gr.Video()
download_button = gr.DownloadButton(label="Download generated video")

demo = gr.Interface(
    fn=gui_inference_single,
    inputs=[input_video, audio_input, hyper_batch_size_slider],
    outputs=[output_video, download_button],
)

#def inference_single():
#    Hyperlips_executor = Hyperlips(checkpoint_path_BASE=args.checkpoint_path_BASE,
#                                    checkpoint_path_HR=args.checkpoint_path_HR,
#                                    segmentation_path=args.segmentation_path,
##                                    face_enhancement_path = args.face_enhancement_path,
 #                                   gpu_id = args.gpu_id,
 #                                   window =args.filter_window,
#                                    hyper_batch_size=args.hyper_batch_size,
#                                    img_size = args.img_size,
#                                    resize_factor = args.resize_factor,
#                                    pad = args.pads)
#    Hyperlips_executor._HyperlipsLoadModels()
#    Hyperlips_executor._HyperlipsInference(args.face,args.audio,args.outfile)


if __name__ == '__main__':
    demo.launch()
