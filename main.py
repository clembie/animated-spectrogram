import configparser
import math
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import sys
import cv2
import subprocess
import shutil
import os


def create_empty_image(bg_color, size_x=1920, size_y=1080):
    """
    This returns the array on which will be drawn.
    """
    bg = np.array(bg_color, dtype=np.uint8)
    img = bg * np.ones((size_y, size_x, 3),
            dtype=np.uint8) * np.ones((size_y, size_x, 1), dtype=np.uint8)
    return img

def get_config(filename):
    """
    All settings are stored in an external text file.
    """
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def get_log_axis_positions(frequencies):
    """
    Returns an array containing the logarithmically
    distributed frequency positions
    on a scale ranging from 0 to 1. Logarithmic scales
    are often helpful to visualize musical pitches.
    """
    axis_positions = []
    max_freq = frequencies[-1]
    total_width = abs(math.log(frequencies[1] / max_freq))
    for f in frequencies[1:]:
        pos = ( math.log(f / max_freq) + total_width ) / total_width
        axis_positions.append(pos)
    return axis_positions

def get_lin_axis_positions(frequencies):
    """
    Returns an array containing the linearly
    distributed frequency positions
    on a scale ranging from 0 to 1.
    """
    axis_positions = []
    max_freq = frequencies[-1]
    for f in frequencies[1:]:
        pos = f / max_freq
        axis_positions.append(pos)
    return axis_positions

def main():
    delete_and_create_folders()
    config = get_config("./options.cfg")["DEFAULT"]

    filename = config["input_file"]
    # stereo_file = config["stereo_file"]
    image_height = int(float(config["image_height"]) * int(config["video_size_y"]))
    windows_length = float(config["windows_length"])
    saturation_value = float(config["saturation_value"])
    crop_bottom = float(config["crop_bottom"])
    crop_top = float(config["crop_top"])
    bg_color = get_color_from_string(config["bg_color"])
    color_active = get_color_from_string(config["color_active"])
    color_silent = get_color_from_string(config["color_silent"])

    sample_rate, data = wavfile.read(filename)
    block_size = int(sample_rate * windows_length)

    f_transforms = []

    mode = "mono"
    print("Create animated spectogram for " + filename + ".")
    print("WARNING: THIS PROGRAM WORKS ONLY WITH MONO WAVE FILES.")
    if mode == "mono":
        no_of_blocks = int( len(data) / float(block_size) )
        audio_duration = len(data) / float(sample_rate)
        T = 1.0 / float(sample_rate)
        max_f = 1.0 / (2.0*T)
        delta_f = max_f / (block_size * 0.5)
        freqs = [i * delta_f for i in range(block_size/2)]
        #axis_positions = get_log_axis_positions(freqs)
        axis_positions = get_lin_axis_positions(freqs)
        max_f_value = 0
        for i in range(no_of_blocks):
            start_index = i * block_size
            end_index = start_index + block_size

            block = data[start_index:end_index]
            f_transform = scipy.fftpack.fft(block) #convert WAV to FFT
            f_transform = abs(f_transform[0:len(f_transform)/2])
            #f_transform = np.log(f_transform)

            f_transforms.append(f_transform)

            if max(f_transform) > max_f_value:
                max_f_value = max(f_transform)

            with open("./single_ffts/%06i.dat" % i, "w") as outfile:
                for index, f in enumerate(f_transform[1:]):
                    outfile.write("%f %f\n" % (axis_positions[index], f))

        delta_pos_in_image = 1.0 / float(image_height)
        positions_in_image = [(i + 1.0) * delta_pos_in_image for i in range(image_height)]

        image_columns = []
        for transform in f_transforms:
            values_in_image = [0.0] * image_height
            for p, v in zip(axis_positions, transform):
                position_in_image = int(p / delta_pos_in_image) - 1
                if position_in_image < 0:
                    position_in_image = 0
                if v > values_in_image[position_in_image]:
                    # multiple values may be mapped to the same pixel
                    # show only the largest value
                    values_in_image[position_in_image] = v
            values_in_image[0] = 0.0
            image_columns.append(values_in_image)

        img_active = create_empty_image([0,0,0], size_x=len(f_transforms), size_y=image_height)
        img_silent = create_empty_image([0,0,0], size_x=len(f_transforms), size_y=image_height)

        for index_column, column in enumerate(image_columns):
            for index_v, v in enumerate(column):
                img_active[image_height-1-index_v][index_column] = value_to_color(v, saturation_value*max_f_value, bg_color, color_active)
                img_silent[image_height-1-index_v][index_column] = value_to_color(v, saturation_value*max_f_value, bg_color, color_silent)

        prepared_images = []
        for img in [img_active, img_silent]:
            img = img[int((1.0-crop_top) * image_height) : int((1.0-crop_bottom) * image_height)][:] # crop
            img = cv2.resize(img, ( int(img.shape[1]*1.0), image_height))
            prepared_images.append(img)

        cv2.imwrite("./output/img_active.png", prepared_images[0])
        cv2.imwrite("./output/img_silent.png", prepared_images[1])

    write_video(config, audio_duration, prepared_images)

def get_color_from_string(color_str):
    """
    This converts the colors from the options file
    to a list of ints: [b,g,r].
    """
    return [int(c) for c in color_str.split(",")]

def resize_image_to_height(img, new_height):
    """
    This preserves the aspect ratio.
    """
    new_width = round(new_height * img.shape[1] / float(img.shape[0]))
    img_resized = cv2.resize(img, (int(new_width), new_height))
    return img_resized

def resize_image_to_width(img, new_width):
    """
    This preserves the aspect ratio.
    """
    new_height = round(new_width * img.shape[0] / float(img.shape[1]))
    img_resized = cv2.resize(img, (new_width, int(new_height)))
    return img_resized

def value_to_color(value, max_v, bg_color, max_color):
    color = []
    for i in range(3):
        degree = min(1.0,value/max_v)
        channel_diff = max_color[i] - bg_color[i]
        color_channel_value = bg_color[i] + degree * channel_diff
        color.append(color_channel_value)
    return color

def print_progress(msg, current, total):
    """
    This keeps the output on the same line.
    """
    text = "\r" + msg + " {:9.1f}/{:.1f}".format(current, total)
    sys.stdout.write(text)
    sys.stdout.flush()

def write_video(config, audio_duration, prepared_images):
    print("Write video. Length of audio: " + str(audio_duration) + " s")
    frame_rate = float(config["frame_rate"])
    waiting_time_before_end = float(config["waiting_time_before_end"])
    start_time = float(config["start_time"])
    highlight_time = float(config["highlight_time"])
    time_before_current = float(config["time_before_current"])
    time_after_current = float(config["time_after_current"])
    video_size_x = int(config["video_size_x"])
    video_size_y = int(config["video_size_y"])
    prepared_images_width = prepared_images[0].shape[1]
    prepared_images_height = prepared_images[0].shape[0]
    bg_color = get_color_from_string(config["bg_color"])
    y_offset = int( (video_size_y - prepared_images_height) * 0.5 )
    on_index = 0
    off_index = 1

    if config["end_time"] == "auto":
        end_time = audio_duration + waiting_time_before_end
    else:
        end_time = float(config["end_time"])

    delta_pixel_time = (time_before_current + time_after_current) / float(video_size_x - 1)
    time = start_time
    img_index = 0
    dt = 1.0 / frame_rate
    while time < end_time:
        time_left = time - time_before_current
        time_right = time + time_after_current
        img = create_empty_image(bg_color, video_size_x, video_size_y)
        # I guess this can be sped up.
        for ix in range(video_size_x):
            pixel_time = time_left + ix * delta_pixel_time
            distance_from_time = abs(pixel_time - time)
            # active = pixel_time >= time and pixel_time < time + highlight_time
            # active_degree = 1.0 - distance_from_time / highlight_time # used to blend images
            active = distance_from_time < 0.5 * highlight_time
            active_degree = 1.0 - distance_from_time / ( highlight_time * 0.5 ) # used to blend images
            if pixel_time >= 0.0 and pixel_time < audio_duration:
                prepared_image_column = int(pixel_time / audio_duration * prepared_images_width)
                y_insert_start = video_size_y - y_offset - prepared_images_height
                y_insert_end = video_size_y  - y_offset
                if active:
                    colors_active = prepared_images[on_index][0:prepared_images_height, prepared_image_column]
                    colors_silent = prepared_images[off_index][0:prepared_images_height, prepared_image_column]
                    mixed = cv2.addWeighted(colors_active, active_degree, colors_silent, 1.0 - active_degree, 0.0)
                    img[y_insert_start:y_insert_end,ix] = mixed
                else:
                    y_insert_start = video_size_y - y_offset - prepared_images_height
                    y_insert_end = video_size_y  - y_offset
                    img[y_insert_start:y_insert_end,ix] = prepared_images[off_index][0:prepared_images_height, prepared_image_column]

        cv2.imwrite("./tmp_images/%08i.png" % img_index, img)
        time += dt
        img_index += 1
        print_progress("Current time:", time, end_time)
    print("")
    run_ffmpeg(frame_rate, video_size_x, video_size_y)
    make_stereo(config["stereo_file"])
    shutil.rmtree("./tmp_images/")

def run_ffmpeg(frame_rate, size_x, size_y):
    """
    Convert all images into a video.
    """
    call_list = []
    call_list.append("ffmpeg")
    call_list.append("-r")
    call_list.append("{:f}".format(frame_rate))
    call_list.append("-f")
    call_list.append("image2")
    call_list.append("-s")
    call_list.append("{:d}x{:d}".format(size_x, size_y))
    call_list.append("-i")
    call_list.append("./tmp_images/%08d.png")
    call_list.append("-vcodec")
    call_list.append("libx264")
    call_list.append("-crf")
    call_list.append("25")
    call_list.append("-pix_fmt")
    call_list.append("yuv420p")
    call_list.append("./output/final.mp4")
    subprocess.call(call_list)

def make_stereo(stereo_file):
    """
    Create a version with stereo sound from supplied stereo audio file
    ffmpeg -i video.mp4 -i audio.wav -map 0:v -map 1:a -c:v copy -shortest output.mp4
    """
    call_list = []
    call_list.append("ffmpeg")
    call_list.append("-i")
    call_list.append("./output/final.mp4")
    call_list.append("-i")
    call_list.append("{:s}".format(stereo_file))
    call_list.append("-map")
    call_list.append("0:v")
    call_list.append("-map")
    call_list.append("1:a")
    call_list.append("-c:v")
    call_list.append("copy")
    call_list.append("-shortest")
    call_list.append("./output/final_st.mp4")
    subprocess.call(call_list)

def delete_and_create_folders():
    """
    Clean everything up first.
    """
    foldernames = ["./single_ffts", "./output", "./tmp_images"]
    for f in foldernames:
        if os.path.isdir(f):
            shutil.rmtree(f)
        os.mkdir(f)


if __name__ == '__main__':
    main()
