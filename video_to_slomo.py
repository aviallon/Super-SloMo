#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
import platform
from tqdm import tqdm
from pathlib import Path
import subprocess
import math

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="", help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, required=True, help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=None, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, required=True, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--vcodec", type=str, default="libx264", help='Specify a codec different than default', choices=['ffvhuff', 'libx264'])
parser.add_argument("--output", type=str, default="output.mkv", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()


def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if args.fps is not None and (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    if ".mkv" not in args.output:
        error = "output needs to have mkv container"
    return error

def extract_frames(ffmpeg_path, video, outDir):
    """
    Converts the `video` to images.

    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """
    error = ""

    ffmpeg_command = '{} -hide_banner -i {} -vsync 0 {}/%06d.png'.format(ffmpeg_path, video, outDir)

    print(ffmpeg_command)
    retn = os.system(ffmpeg_command)
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
        return error

    ffmpeg_command = '{} -hide_banner -i {} -c:a copy -map 0:a? {}/../audio.mkv'.format(ffmpeg_path, video, outDir)
    print("Extracting audio:\n", ffmpeg_command)
    retn = os.system(ffmpeg_command)
    if retn:
        error = "Error extracting audio from file:{}. Exiting.".format(video)
    return error

def create_video(ffmpeg_path, dir, speed_factor):

    error = ""
    
    codec_args = ""
    if args.vcodec == 'libx264':
        codec_args = "-preset:v veryfast -crf 18 -pix_fmt yuv420p -tune:v film -profile:v high"
        
    acodec = "copy"
    acodec_args = ""
    if speed_factor != 1:
        acodec = "libopus"
        acodec_args = f"-b:a 128k -frame_duration 120 -cutoff 20000 -filter:a "
        acodec_filters = []
        speed_nums = 1
        remain = speed_factor
        speed_quantum = 0.5
        if speed_factor < 0.5:
            speed_quantum = 0.5
        elif speed_factor > 2:
            speed_quantum = 2
            
            
        if speed_nums != 1:
            speed_nums = int(math.log(speed_factor, speed_quantum))
            remain = speed_factor / speed_quantum**speed_nums
            
            acodec_filters += [f"atempo={speed_quantum}" for i in range(speed_nums)]
            
        if remain != 0:
            acodec_filters += [f"atempo={remain}"]
                    
        acodec_args += ",".join(acodec_filters)
    
    ffmpeg_command = f'{ffmpeg_path} -hide_banner -r {args.fps} -i {dir}/%d.png -i {dir}/../audio.mkv -acodec {acodec} {acodec_args} -vcodec {args.vcodec} {codec_args} -map 0:v:0 -map 1:a:0? {Path(args.output).resolve()}'
    
    print(ffmpeg_command)
    retn = os.system(ffmpeg_command)
    if retn:
        error = "Error creating output video. Exiting."
    return error


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    # Create extraction folder and extract frames
    
    IS_WINDOWS = 'Windows' == platform.system()


    ffmpeg_path = "ffmpeg"
    
    extractionDir = "tmpSuperSloMo"
    if not IS_WINDOWS:
        extractionDir = os.path.join("/var/tmp", extractionDir)
        
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
        
    os.mkdir(extractionDir)
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        # ctypes.windll only exists on Windows
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)
        
        ffmpeg_path = os.path.join(args.ffmpeg_dir, "ffmpeg")

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
    
    
    
    # Get input video fps
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    
    ffprobe_command = [ffprobe_path, "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", str(Path(args.video).resolve())]
    print(" ".join(ffprobe_command), flush=True)
    res = subprocess.run(ffprobe_command, stdout=subprocess.PIPE)
    
    #print("OUTPUT:", res.stdout)
    
    ffprobe_output = res.stdout.decode().strip().split("/")
    input_fps = float(ffprobe_output[0]) / float(ffprobe_output[1])
    
    if args.fps is None:
        args.fps = input_fps * args.sf
        print(f"No fps specified, default to {args.fps}")

    
    speed_factor = args.fps / (input_fps * args.sf)
    
    if abs(1 - speed_factor) < 1e-8:
        speed_factor = 1
        
    print(f"Speed factor: {speed_factor} (input video FPS: {input_fps})")
    
    
    error = extract_frames(ffmpeg_path, Path(args.video).resolve(), extractionPath)
    if error:
        print(error)
        exit(1)

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = float(intermediateIndex) / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0

                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".png"))
                frameCounter += 1

            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)


    # Generate video from interpolated frames
    create_video(ffmpeg_path, outputPath, speed_factor)

    # Remove temporary files
    rmtree(extractionDir)

    exit(0)

main()
