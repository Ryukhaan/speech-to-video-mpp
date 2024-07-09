from Lipsreenact import LipsReenact as Reenacter
import argparse
import os
import gradio as gr
import torch
from transformers import pipeline, VitsModel, AutoTokenizer
import scipy

#from TTS.api import TTS
#tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using HyperLipsBase or HyperLipsHR models')
parser.add_argument('--checkpoint_path_BASE', type=str,help='Name of saved HyperLipsBase checkpoint to load weights from', default="checkpoints/hyperlipsbase_lrs2.pth")
parser.add_argument('--checkpoint_path_HR', type=str,help='Name of saved HyperLipsHR checkpoint to load weights from', default="checkpoints/hyperlipshr_lrs2_512.pth")
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

options = dict()

def gui_inference_single(face, audio):
    progress = gr.Progress()
    progress(0, desc="Starting...")
    print(options["hyper_batch_size"].value)
    executor = Reenacter(checkpoint_path_BASE=options["checkpoint_autoencoder"].value['path'],
                         checkpoint_path_HR=options["model_hr"].value['path'],
                         segmentation_path=options["face_segmentation"].value['path'],
                         face_enhancement_path=options["face_enhancement"].value['path'],
                         gpu_id=args.gpu_id,
                         window=args.filter_window,
                         hyper_batch_size=options["hyper_batch_size"].value,
                         img_size=args.img_size,
                         resize_factor=args.resize_factor,
                         pad=args.pads)
    executor._LoadModels()
    executor._Inference(face, audio, args.outfile, progress)
    return args.outfile, args.outfile

def get_tts(text, progress=gr.Progress()):

    progress(0., "Loading TTS model")
    # generate speech by cloning a voice using default settings
    #tts.tts_to_file(
    #    text=text,
    #    file_path="./bark_out.wav",
    #    #speaker_wav="/path/to/target/speaker.wav",
    #    language="fr")
    model = VitsModel.from_pretrained("facebook/mms-tts-fra")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")
    inputs = tokenizer(text, return_tensors="pt")

    progress(0.2, "Generate Audio")
    with torch.no_grad():
        output = model(**inputs).waveform

    progress(0.8, "Write File")
    scipy.io.wavfile.write("./bark_out.wav", rate=model.config.sampling_rate, data=output.cpu().numpy().T)
    return "./bark_out.wav"

demo = gr.Blocks()

with demo:
    with gr.Row():

        with gr.Column():
            hyper_batch_size_slider = gr.Slider(label="Batch Size",
                                                value=32,
                                                minimum=4,
                                                maximum=128,
                                                step=4)
            options["hyper_batch_size"] = hyper_batch_size_slider

            #face_enhancement_checkbox = gr.Checkbox(True,
            #                                        label="Use Face Enhancement",
            #                                        container=True)
            #options["face_enhancement"] = face_enhancement_checkbox

            autoencoder_filepath = gr.File(label="Path to AutoEncoder Lips-Sync Network. default: checkpoints/hyperlipsbase_lrs2.pth",
                                           value="./checkpoints/hyperlipsbase_lrs2.pth",
                                           interactive=True)
            options["checkpoint_autoencoder"] = autoencoder_filepath

            own_srmodel_filepath = gr.File(label="Path to Own Super Resolution Network. default: checkpoints/hyperlipshr_lrs2_128.pth",
                                        value="./checkpoints/hyperlipshr_lrs2_128.pth",
                                        interactive=True)
            options["model_hr"] = own_srmodel_filepath

            gpgfpgan_filepath = gr.File(label="Path to Super Resolution Network. default: checkpoints/GFPGANv1.3.pth",
                                        value="./checkpoints/GFPGANv1.3.pth",
                                        interactive=True)
            options["face_enhancement"] = gpgfpgan_filepath


            face_segmentation_filepath = gr.File(label="Checkpoint of segmentation network. default: checkpoints/face_segmentation.pth",
                                                 value="./checkpoints/face_segmentation.pth",
                                                 interactive=True)
            options["face_segmentation"] = face_segmentation_filepath


        with gr.Column():


            input_video = gr.Video(label="Upload a video",
                                   format="mp4")
            reenactment_button = gr.Button("Reenact Video !")

            audio_input = gr.Audio(label="Upload an audio",
                                   type='filepath',
                                   format="wav")

            text_input = gr.Textbox(label="Type your text")
            generate_tts_button = gr.Button(value="Generate TTS")

            generate_tts_button.click(get_tts, inputs=[text_input], outputs=[audio_input])

        with gr.Column():
            output_video = gr.Video()
            download_button = gr.DownloadButton(label="Download generated video")

            reenactment_button.click(gui_inference_single,
                                     inputs=[input_video, audio_input],
                                     outputs=[output_video, download_button])

if __name__ == '__main__':
    demo.launch(server_port=8008)
