import os
import tempfile
import torch
import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
import time
import gc
import argparse
import numpy as np
import locale


# 检测设备类型并设置默认精度
def get_default_precision():
    """根据设备类型返回最合适的默认精度"""
    if torch.cuda.is_available():
        # CUDA GPU: float16 性能最好
        return "float16"
    elif torch.backends.mps.is_available():
        # Apple Silicon MPS: float16 性能较好
        return "float16"
    else:
        # CPU: float32 稳定性最好
        return "float32"

# 解析命令行参数
parser = argparse.ArgumentParser(description="HeartMuLa Music Generator with precision options")
parser.add_argument("--precision", type=str, default=get_default_precision(), choices=["float32", "float16", "bfloat16", "int8", "int4"],
                    help=f"Model precision (default: auto-detected based on device)")
parser.add_argument("--offline", action="store_true", default=True,
                    help="Use offline mode (default: True)")
parser.add_argument("--model-dir", type=str, default="./ckpt/heartmula_models",
                    help="Model directory (default: ./ckpt/heartmula_models)")
parser.add_argument("--lazy-load", type=lambda x: x.lower() == "true", default=True,
                    help="Lazy load model components (default: True)")
args = parser.parse_args()

# Log precision selection logic
print("=== Precision Configuration ===")
print(f"Detected device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
print(f"Default precision for device: {get_default_precision()}")
print(f"Command line precision argument: {args.precision}")
print(f"Selected precision: {args.precision}")
print("=== Precision Configuration Complete ===")

# Detect system language
try:
    system_lang = locale.getdefaultlocale()[0]
    if system_lang and 'zh' in system_lang:
        current_lang = 'zh'
    else:
        current_lang = 'en'
except:
    current_lang = 'en'

print(f"Detected system language: {current_lang}")

# Load lyrics and tags based on language
print(f"Current working directory: {os.getcwd()}")
demo_data_dir = os.path.join(os.getcwd(), "demo-data")
print(f"Demo data directory: {demo_data_dir}")
lyrics_file = os.path.join(demo_data_dir, f"lyric-{current_lang}.txt")
tags_file = os.path.join(demo_data_dir, f"tags-{current_lang}.txt")
print(f"Lyrics file path: {lyrics_file}")
print(f"Tags file path: {tags_file}")

# Check if files exist
print(f"Lyrics file exists: {os.path.exists(lyrics_file)}")
print(f"Tags file exists: {os.path.exists(tags_file)}")

# Read lyrics
try:
    with open(lyrics_file, 'r', encoding='utf-8') as f:
        EXAMPLE_LYRICS = f.read()
    print(f"Loaded lyrics from {lyrics_file}")
except Exception as e:
    print(f"Error loading lyrics file: {str(e)}")
    # Default lyrics
    EXAMPLE_LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Outro]
Just another day
Every single day"""

# Read tags
try:
    with open(tags_file, 'r', encoding='utf-8') as f:
        EXAMPLE_TAGS = f.read().strip()
    print(f"Loaded tags from {tags_file}")
except Exception as e:
    print(f"Error loading tags file: {str(e)}")
    # Default tags
    EXAMPLE_TAGS = "piano,happy,uplifting,pop"

# Download models from HuggingFace Hub
def download_models(offline=False, model_dir=None):
    """Download all required model files from HuggingFace Hub or use local files in offline mode."""
    if model_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("/tmp"))
        model_dir = os.path.join(cache_dir, "heartmula_models")

    if not os.path.exists(model_dir):
        if offline:
            raise FileNotFoundError(f"Model directory {model_dir} does not exist in offline mode")
        os.makedirs(model_dir, exist_ok=True)

    if offline:
        # Check if all required directories exist
        required_dirs = [
            os.path.join(model_dir, "HeartMuLa-oss-3B"),
            os.path.join(model_dir, "HeartCodec-oss")
        ]
        required_files = [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(model_dir, "gen_config.json")
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory {dir_path} does not exist in offline mode")
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file {file_path} does not exist in offline mode")
        
        print("Using offline mode, loading models from local directory...")
        return model_dir

    # Download HeartMuLaGen (tokenizer and gen_config)
    print("Downloading HeartMuLaGen files...")
    for filename in ["tokenizer.json", "gen_config.json"]:
        hf_hub_download(
            repo_id="HeartMuLa/HeartMuLaGen",
            filename=filename,
            local_dir=model_dir,
        )

    # Download HeartMuLa-oss-3B
    print("Downloading HeartMuLa-oss-3B...")
    snapshot_download(
        repo_id="HeartMuLa/HeartMuLa-oss-3B",
        local_dir=os.path.join(model_dir, "HeartMuLa-oss-3B"),
    )

    # Download HeartCodec-oss
    print("Downloading HeartCodec-oss...")
    snapshot_download(
        repo_id="HeartMuLa/HeartCodec-oss",
        local_dir=os.path.join(model_dir, "HeartCodec-oss"),
    )

    print("All models downloaded successfully!")
    return model_dir

# Get device and dtype based on precision
def get_device_and_dtype(precision):
    """Get appropriate device and dtype based on precision and hardware availability."""
    print("=== Device and Dtype Configuration ===")
    print(f"Requested precision: {precision}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Selected device: {device}")
        # 根据精度选择 dtype
        if precision == "float32":
            dtype = torch.float32
            print("Selected dtype: torch.float32")
        elif precision == "float16":
            dtype = torch.float16
            print("Selected dtype: torch.float16")
        elif precision == "bfloat16":
            try:
                # 检查 GPU 是否支持 bfloat16
                test_tensor = torch.tensor([1.0], device=device, dtype=torch.bfloat16)
                dtype = torch.bfloat16
                print("Selected dtype: torch.bfloat16 (GPU supported)")
            except Exception as e:
                print(f"bfloat16 not supported on this GPU, falling back to float16: {str(e)}")
                dtype = torch.float16
                print("Fallback dtype: torch.float16")
        elif precision == "int8" or precision == "int4":
            # 对于量化精度，先使用 float16 作为基础
            dtype = torch.float16
            print(f"Selected dtype for {precision} precision: torch.float16 (base dtype)")
        else:
            dtype = torch.float16
            print(f"Invalid precision '{precision}', falling back to torch.float16")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Selected device: {device} (Apple Silicon)")
        # MPS 精度支持
        if precision == "float32":
            dtype = torch.float32
            print("Selected dtype: torch.float32")
        elif precision == "float16":
            # MPS 对 float16 的支持有限，使用 float32 以避免 dtype 不匹配错误
            print("float16 not fully supported on MPS, falling back to float32")
            dtype = torch.float32
            print("Fallback dtype: torch.float32")
        elif precision == "bfloat16":
            # MPS 对 bfloat16 的支持有限
            print("bfloat16 not fully supported on MPS, falling back to float32")
            dtype = torch.float32
            print("Fallback dtype: torch.float32")
        elif precision == "int8" or precision == "int4":
            # 对于量化精度，使用 float32 作为基础以避免 dtype 不匹配错误
            dtype = torch.float32
            print(f"Selected dtype for {precision} precision: torch.float32 (base dtype)")
        else:
            dtype = torch.float32
            print(f"Invalid precision '{precision}', falling back to torch.float32")
    else:
        device = torch.device("cpu")
        # CPU 上使用 float32 以确保稳定性
        dtype = torch.float32
        print(f"CUDA and MPS not available, using CPU with dtype: torch.float32")
    
    # 检查 GPU 内存
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"GPU memory: {total_mem:.2f} GB")
        if total_mem < 8:
            print("Warning: GPU memory may be insufficient for the model")
    
    print(f"=== Final Configuration: Device={device}, Dtype={dtype} ===")
    return device, dtype

# Load model with specified precision
def load_model(model_dir, precision, lazy_load=False):
    """Load model with specified precision."""
    from heartlib import HeartMuLaGenPipeline
    
    # 获取设备和数据类型
    device, dtype = get_device_and_dtype(precision)
    
    print(f"Loading pipeline on {device} with {precision} precision...")
    
    try:
        # 加载模型
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_dir,
            device=device,
            dtype=dtype,
            version="3B",
            lazy_load=lazy_load,
        )
        
        # 开启评估模式
        pipe.mula.eval()
        pipe.codec.eval()
        
        # 显存检查
        if device.type == "cuda":
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f">>> Current static VRAM usage: {vram_allocated:.2f} GB")
        
        print("Pipeline loaded successfully!")
        return pipe, device, dtype
        
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        # 尝试回退到 float16
        print("Attempting to load with float16 precision...")
        try:
            # 确保 device 是 torch.device 类型
            if torch.cuda.is_available():
                fallback_device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                fallback_device = torch.device("mps")
            else:
                fallback_device = torch.device("cpu")
            
            # 为不同设备选择合适的 dtype
            if fallback_device.type == "cpu":
                fallback_dtype = torch.float32
            else:
                fallback_dtype = torch.float16
                
            pipe = HeartMuLaGenPipeline.from_pretrained(
                model_dir,
                device=fallback_device,
                dtype=fallback_dtype,
                version="3B",
                lazy_load=lazy_load,
            )
            pipe.mula.eval()
            pipe.codec.eval()
            print(f"Pipeline loaded successfully with {fallback_dtype}!")
            return pipe, fallback_device, fallback_dtype
        except Exception as e2:
            print(f"Error loading pipeline with fallback precision: {e2}")
            # 尝试回退到 CPU
            print("Falling back to CPU...")
            cpu_device = torch.device("cpu")
            pipe = HeartMuLaGenPipeline.from_pretrained(
                model_dir,
                device=cpu_device,
                dtype=torch.float32,
                version="3B",
                lazy_load=lazy_load,
            )
            pipe.mula.eval()
            pipe.codec.eval()
            print("Pipeline loaded successfully on CPU!")
            return pipe, cpu_device, torch.float32

# Download models
model_dir = download_models(offline=args.offline, model_dir=args.model_dir)

# Load model with specified precision
pipe, device, dtype = load_model(model_dir, args.precision, lazy_load=args.lazy_load)

# 检查 PyTorch 版本
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# 检查是否需要使用替代方法保存音频
use_alternative_save = False
if torch_version.startswith('2.9.1'):
    print("Detected PyTorch 2.9.1, will use alternative save method to avoid torchcodec dependency")
    use_alternative_save = True

# 尝试直接使用 soundfile 保存，避免 torchcodec 依赖
try:
    import soundfile as sf
    has_soundfile = True
except ImportError:
    has_soundfile = False
    print("Warning: soundfile not available, will use default save method")

# 尝试使用内置的 wave 模块作为备选
try:
    import wave
    import struct
    has_wave = True
except ImportError:
    has_wave = False
    print("Warning: wave module not available")

def generate_music(
    lyrics: str,
    tags: str,
    max_duration_seconds: int,
    temperature: float,
    topk: int,
    cfg_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate music from lyrics and tags."""
    if not lyrics.strip():
        raise gr.Error("Please enter some lyrics!")

    if not tags.strip():
        raise gr.Error("Please enter at least one tag!")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    timestamp = int(time.time())
    output_filename = f"output_{timestamp}.wav"
    output_path = os.path.join(output_dir, output_filename)

    max_audio_length_ms = max_duration_seconds * 1000

    # Start logging
    start_time = time.time()
    print(f"=== Music Generation Started ===")
    print(f"Lyrics length: {len(lyrics)} characters")
    print(f"Tags: {tags}")
    print(f"Max duration: {max_duration_seconds} seconds")
    print(f"Temperature: {temperature}")
    print(f"Top-K: {topk}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Output path: {output_path}")
    print(f"Starting generation process...")

    # 记录实际的保存路径
    actual_save_path = None

    try:
        # 强制清理
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # 创建进度条
            with tqdm(total=100, desc="Generating music", unit="%") as pbar:
                # 更新进度条
                pbar.update(10)
                print("[10%] Initializing model...")
                
                if use_alternative_save and has_soundfile:
                    print("Using alternative save method with soundfile...")
                    
                    # 修改 pipe 的 postprocess 方法，使其返回 wav 数据
                    original_postprocess = pipe.postprocess
                    
                    # 保存 wav 数据的字典
                    wav_data_container = {"data": None}
                    
                    def custom_postprocess(model_outputs, save_path):
                        frames = model_outputs["frames"].to(pipe.codec_device)
                        wav = pipe.codec.detokenize(frames)
                        pipe._unload()
                        wav_data_container["data"] = wav.to(torch.float32).cpu()
                        # 不调用 torchaudio.save
                    
                    # 替换 postprocess 方法
                    pipe.postprocess = custom_postprocess
                    
                    try:
                        # 执行生成
                        pbar.update(30)
                        print("[40%] Processing lyrics and tags...")
                        
                        # 记录生成开始时间
                        generation_start_time = time.time()
                        
                        result = pipe(
                            {
                                "lyrics": lyrics,
                                "tags": tags,
                            },
                            max_audio_length_ms=max_audio_length_ms,
                            save_path=None,  # 这个值会被覆盖，但我们的自定义方法不使用它
                            topk=topk,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                        )
                        
                        # 记录生成结束时间
                        generation_end_time = time.time()
                        generation_duration = generation_end_time - generation_start_time
                        
                        pbar.update(30)
                        print("[70%] Generating audio frames...")
                        
                        # 恢复原始 postprocess 方法
                        pipe.postprocess = original_postprocess
                        
                        # 保存 wav 数据
                        wav_data = wav_data_container["data"]
                        if wav_data is not None:
                            # 确保目录存在
                            save_dir = os.path.dirname(output_path)
                            if save_dir and not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                                print(f"Created directory: {save_dir}")
                            
                            # 尝试保存为 WAV 格式
                            try:
                                if has_soundfile:
                                    try:
                                        sf.write(output_path, wav_data.numpy(), 48000, format='WAV', subtype='PCM_16')
                                        print(f"Audio saved successfully with soundfile: {output_path}")
                                        actual_save_path = output_path
                                    except Exception as e:
                                        print(f"Error saving WAV with soundfile: {str(e)}")
                                        print("Attempting to save with wave module...")
                                        raise
                                else:
                                    raise Exception("soundfile not available")
                            except Exception as e:
                                if has_wave:
                                    print("Using wave module to save audio...")
                                    # 使用内置的 wave 模块保存
                                    try:
                                        # 确保音频数据是 16 位整数
                                        audio_data = wav_data.numpy()
                                        # 归一化到 [-1, 1]
                                        audio_data = audio_data / np.max(np.abs(audio_data))
                                        # 转换为 16 位整数
                                        audio_data = np.int16(audio_data * 32767)
                                        
                                        # 打开 WAV 文件
                                        with wave.open(output_path, 'w') as wf:
                                            wf.setnchannels(1)  # 单声道
                                            wf.setsampwidth(2)  # 16 位
                                            wf.setframerate(48000)  # 48kHz
                                            wf.writeframes(audio_data.tobytes())
                                        print(f"Audio saved successfully with wave module: {output_path}")
                                        actual_save_path = output_path
                                    except Exception as e2:
                                        print(f"Error saving audio with wave module: {str(e2)}")
                                        raise
                                else:
                                    print("No alternative save method available")
                                    raise
                        else:
                            raise Exception("No wav data generated")
                    finally:
                        # 恢复原始 postprocess 方法
                        pipe.postprocess = original_postprocess
                else:
                    # 使用默认保存方法
                    try:
                        pbar.update(30)
                        print("[40%] Processing lyrics and tags...")
                        
                        # 记录生成开始时间
                        generation_start_time = time.time()
                        
                        result = pipe(
                            {
                                "lyrics": lyrics,
                                "tags": tags,
                            },
                            max_audio_length_ms=max_audio_length_ms,
                            save_path=output_path,
                            topk=topk,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                        )
                        
                        # 记录生成结束时间
                        generation_end_time = time.time()
                        generation_duration = generation_end_time - generation_start_time
                        
                        pbar.update(30)
                        print("[70%] Generating audio frames...")
                        
                        print("Audio saved successfully with default method!")
                        actual_save_path = output_path
                    except ImportError as e:
                        if "torchcodec" in str(e) and has_soundfile:
                            print(f"Error saving audio: {str(e)}")
                            print("Attempting to save with soundfile...")
                            
                            # 修改 pipe 的 postprocess 方法，使其返回 wav 数据
                            original_postprocess = pipe.postprocess
                            
                            # 保存 wav 数据的字典
                            wav_data_container = {"data": None}
                            
                            def custom_postprocess(model_outputs, save_path):
                                frames = model_outputs["frames"].to(pipe.codec_device)
                                wav = pipe.codec.detokenize(frames)
                                pipe._unload()
                                wav_data_container["data"] = wav.to(torch.float32).cpu()
                                # 不调用 torchaudio.save
                            
                            # 替换 postprocess 方法
                            pipe.postprocess = custom_postprocess
                            
                            try:
                                # 执行生成
                                pbar.update(30)
                                print("[40%] Processing lyrics and tags...")
                                
                                # 记录生成开始时间
                                generation_start_time = time.time()
                                
                                result = pipe(
                                    {
                                        "lyrics": lyrics,
                                        "tags": tags,
                                    },
                                    max_audio_length_ms=max_audio_length_ms,
                                    save_path=None,  # 这个值会被覆盖，但我们的自定义方法不使用它
                                    topk=topk,
                                    temperature=temperature,
                                    cfg_scale=cfg_scale,
                                )
                                
                                # 记录生成结束时间
                                generation_end_time = time.time()
                                generation_duration = generation_end_time - generation_start_time
                                
                                pbar.update(30)
                                print("[70%] Generating audio frames...")
                                
                                # 恢复原始 postprocess 方法
                                pipe.postprocess = original_postprocess
                                
                                # 保存 wav 数据
                                wav_data = wav_data_container["data"]
                                if wav_data is not None:
                                    # 确保目录存在
                                    save_dir = os.path.dirname(output_path)
                                    if save_dir and not os.path.exists(save_dir):
                                        os.makedirs(save_dir)
                                        print(f"Created directory: {save_dir}")
                                    
                                    # 尝试保存为 WAV 格式
                                    try:
                                        if has_soundfile:
                                            try:
                                                sf.write(output_path, wav_data.numpy(), 48000, format='WAV', subtype='PCM_16')
                                                print(f"Audio saved successfully with soundfile: {output_path}")
                                                actual_save_path = output_path
                                            except Exception as e:
                                                print(f"Error saving WAV with soundfile: {str(e)}")
                                                print("Attempting to save with wave module...")
                                                raise
                                        else:
                                            raise Exception("soundfile not available")
                                    except Exception as e:
                                        if has_wave:
                                            print("Using wave module to save audio...")
                                            # 使用内置的 wave 模块保存
                                            try:
                                                # 确保音频数据是 16 位整数
                                                audio_data = wav_data.numpy()
                                                # 归一化到 [-1, 1]
                                                audio_data = audio_data / np.max(np.abs(audio_data))
                                                # 转换为 16 位整数
                                                audio_data = np.int16(audio_data * 32767)
                                                
                                                # 打开 WAV 文件
                                                with wave.open(output_path, 'w') as wf:
                                                    wf.setnchannels(1)  # 单声道
                                                    wf.setsampwidth(2)  # 16 位
                                                    wf.setframerate(48000)  # 48kHz
                                                    wf.writeframes(audio_data.tobytes())
                                                print(f"Audio saved successfully with wave module: {output_path}")
                                                actual_save_path = output_path
                                            except Exception as e2:
                                                print(f"Error saving audio with wave module: {str(e2)}")
                                                raise
                                        else:
                                            print("No alternative save method available")
                                            raise
                                else:
                                    raise Exception("No wav data generated")
                            finally:
                                # 恢复原始 postprocess 方法
                                pipe.postprocess = original_postprocess
                        else:
                            raise
                
                # 更新进度条
                pbar.update(30)
                print("[100%] Generation completed!")
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise
    finally:
        # Calculate and log generation time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"=== Music Generation Completed ===")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        
        # 计算生成速度统计信息
        if 'generation_duration' in locals():
            print(f"Actual generation time: {generation_duration:.2f} seconds")
            # 计算每秒处理的音频长度
            audio_duration_seconds = max_duration_seconds
            if audio_duration_seconds > 0:
                generation_speed = audio_duration_seconds / generation_duration
                print(f"Generation speed: {generation_speed:.2f} seconds of audio per second")
        
        # 计算整体速度
        audio_duration_seconds = max_duration_seconds
        if audio_duration_seconds > 0:
            overall_speed = audio_duration_seconds / elapsed_time
            print(f"Overall speed: {overall_speed:.2f} seconds of audio per second")
        
        # 显示实际的保存路径
        if actual_save_path:
            print(f"Generated music saved to: {actual_save_path}")
        else:
            # 检查是否实际保存为 WAV 文件
            if os.path.exists(output_path):
                print(f"Generated music saved to: {output_path}")
            else:
                print(f"Generated music save path: {output_path}")
                print("Note: The actual save path may be different if conversion was needed")

    # Determine the final save path
    final_save_path = actual_save_path if actual_save_path else output_path
    
    # Check if the file actually exists
    if not os.path.exists(final_save_path):
        raise Exception(f"Failed to save audio file: {final_save_path}")
    
    return final_save_path, final_save_path

# Define UI text in both languages
ui_text = {
    'en': {
        'title': 'HeartMuLa Music Generation',
        'description': 'Create music from lyrics and tags with [HeartMuLa](https://github.com/HeartMuLa/heartlib), an open-source music model with multilingual support.',
        'lyrics_label': 'Lyrics',
        'lyrics_placeholder': 'Enter lyrics with structure tags like [Verse], [Chorus], etc.',
        'tags_label': 'Tags',
        'tags_placeholder': 'piano,happy,romantic,synthesizer',
        'tags_info': 'Comma-separated tags describing the music style',
        'advanced_settings': 'Advanced Settings',
        'max_duration_label': 'Max Duration (seconds)',
        'max_duration_info': 'Maximum length of generated audio',
        'temperature_label': 'Temperature',
        'temperature_info': 'Higher = more creative, Lower = more consistent',
        'topk_label': 'Top-K',
        'topk_info': 'Number of top tokens to sample from',
        'cfg_scale_label': 'CFG Scale',
        'cfg_scale_info': 'Classifier-free guidance scale',
        'generate_button': 'Generate Music',
        'generated_music': 'Generated Music',
        'instructions': '## Instructions',
        'instruction_steps': '1. Enter your lyrics with structure tags like `[Verse]`, `[Chorus]`, `[Bridge]`, etc.\n2. Add comma-separated tags describing the music style (e.g., `piano,happy,romantic`)\n3. Adjust generation parameters as needed\n4. Click "Generate Music" and wait for your song!',
        'note': '*Note: Generation can take several minutes depending on the duration and precision.*',
        'tips': '### Tips for Better Results',
        'tip_points': '- Use structured lyrics with section tags\n- Be specific with your style tags\n- Try different temperature values for variety\n- Shorter durations generate faster',
        'speed_optimization': '### Speed Optimization Tips',
        'speed_tips': '- **Shorter Duration:** Reduce max duration for faster generation\n- **Higher Top-K:** Increase top-k value (e.g., 80-100) for faster sampling\n- **Lower Temperature:** Lower temperature (e.g., 0.7-0.9) can speed up generation\n- **Hardware Acceleration:** Ensure CUDA is enabled for GPU acceleration',
        'example_tags': '### Example Tags',
        'instrument_tags': '- **Instruments:** piano, guitar, drums, synthesizer, violin, bass',
        'mood_tags': '- **Mood:** happy, sad, romantic, energetic, calm, melancholic',
        'genre_tags': '- **Genre:** pop, rock, jazz, classical, electronic, folk',
        'tempo_tags': '- **Tempo:** fast, slow, upbeat, relaxed',
        'atmospheric_context_tags': '- **Atmospheric/Context:** wedding, atmospheric, cinematic, lofi, cyberpunk healing dark, bright, nostalgic',
        'vocal_style_tags': '- **Vocal Style:** male_vocal, female_vocal, whisper, powerful, airy',
        'model': '**Model:**',
        'paper': '**Paper:**',
        'code': '**Code:**',
        'license': '*Licensed under Apache 2.0*',
        'download_audio': 'Download Audio',
        'current_environment': 'Current Environment'
    },
    'zh': {
        'title': 'HeartMuLa 音乐生成',
        'description': '使用 [HeartMuLa](https://github.com/HeartMuLa/heartlib) 创建歌词和标签生成的音乐，这是一个支持多语言的开源音乐模型。',
        'lyrics_label': '歌词',
        'lyrics_placeholder': '输入带有结构标签的歌词，如 [Verse], [Chorus] 等',
        'tags_label': '标签',
        'tags_placeholder': 'piano,happy,romantic,synthesizer',
        'tags_info': '用逗号分隔的描述音乐风格的标签',
        'advanced_settings': '高级设置',
        'max_duration_label': '最大时长 (秒)',
        'max_duration_info': '生成音频的最大长度',
        'temperature_label': '温度',
        'temperature_info': '越高 = 越有创意，越低 = 越一致',
        'topk_label': 'Top-K',
        'topk_info': '采样的前 K 个标记数量',
        'cfg_scale_label': 'CFG 缩放',
        'cfg_scale_info': '无分类器引导缩放',
        'generate_button': '生成音乐',
        'generated_music': '生成的音乐',
        'instructions': '## 使用说明',
        'instruction_steps': '1. 输入带有结构标签的歌词，如 `[Verse]`, `[Chorus]`, `[Bridge]` 等\n2. 用逗号分隔的描述音乐风格的标签（例如：`钢琴,欢快,浪漫`）\n3. 根据需要调整生成参数\n4. 点击 "生成音乐" ，等待您的歌曲！',
        'note': '*注意：生成时间可能几分钟或更久，取决于硬件配置、时长和精度等因素。*',
        'tips': '### 获得更好结果：',
        'tip_points': '- 使用带有章节标签的结构化歌词\n- 具体说明您的风格标签\n- 尝试不同的温度值以获得多样性\n- 较短的时长生成更快',
        'speed_optimization': '### 速度优化提示',
        'speed_tips': '- **较短时长：** 减少最大时长以加快生成速度\n- **较高 Top-K：** 增加 top-k 值（例如 80-100）以加快采样速度\n- **较低温度：** 降低温度（例如 0.7-0.9）可以加快生成速度',
        'example_tags': '### 示例标签',
        'instrument_tags': '- **乐器：** piano, guitar, drums, synthesizer, violin, bass',
        'mood_tags': '- **情绪：** happy, sad, romantic, energetic, calm, melancholic',
        'genre_tags': '- **流派：** pop, rock, jazz, classical, electronic, folk',
        'tempo_tags': '- **Tempo：** fast, slow, upbeat, relaxed',
        'atmospheric_context_tags': '- **场景/氛围:** wedding, atmospheric, cinematic, lofi, cyberpunk healing dark, bright, nostalgic',
        'vocal_style_tags': '- **人声特征:** male_vocal, female_vocal, whisper, powerful, airy',
        'model': '**模型：**',
        'paper': '**论文：**',
        'code': '**代码：**',
        'license': '*基于 Apache 2.0 许可证*',
        'download_audio': '下载音频',
        'current_environment': '当前环境'
    }
}

# Get current language text
current_ui_text = ui_text[current_lang]

# Build the Gradio interface
with gr.Blocks(
    title=current_ui_text['title']
) as demo:
    gr.Markdown(
        f"""
        # {current_ui_text['title']}

        {current_ui_text['description']} <small>**{current_ui_text['current_environment']}:** Precision: {args.precision} | PyTorch Version: {torch_version}</small>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=500):
            lyrics_input = gr.Textbox(
                label=current_ui_text['lyrics_label'],
                placeholder=current_ui_text['lyrics_placeholder'],
                lines=9,
                max_lines=14,
                value=EXAMPLE_LYRICS,
                show_label=True,
            )

            tags_input = gr.Textbox(
                label=current_ui_text['tags_label'],
                placeholder=current_ui_text['tags_placeholder'],
                value=EXAMPLE_TAGS,
                info=current_ui_text['tags_info'],
                show_label=True,
            )

            generate_btn = gr.Button(current_ui_text['generate_button'], variant="primary", size="lg")

            with gr.Accordion(current_ui_text['advanced_settings'], open=False):
                max_duration = gr.Slider(
                    minimum=20,
                    maximum=240,
                    value=120,
                    step=10,
                    label=current_ui_text['max_duration_label'],
                    info=current_ui_text['max_duration_info'],
                )

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label=current_ui_text['temperature_label'],
                    info=current_ui_text['temperature_info'],
                )

                topk = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label=current_ui_text['topk_label'],
                    info=current_ui_text['topk_info'],
                )

                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label=current_ui_text['cfg_scale_label'],
                    info=current_ui_text['cfg_scale_info'],
                )

        with gr.Column(scale=1, min_width=500):
            audio_output = gr.Audio(
                label=current_ui_text['generated_music'],
                type="filepath",
                show_label=True,
            )
            
            download_button = gr.DownloadButton(
                label=current_ui_text['download_audio'],
                variant="secondary",
                size="sm"
            )

            gr.Markdown(
                f"""
{current_ui_text['instructions']}
{current_ui_text['instruction_steps']}

{current_ui_text['note']}
                """
            )

            gr.Markdown(
                f"""
{current_ui_text['example_tags']}
{current_ui_text['instrument_tags']}
{current_ui_text['mood_tags']}
{current_ui_text['genre_tags']}
{current_ui_text['tempo_tags']}
{current_ui_text['atmospheric_context_tags']}
{current_ui_text['vocal_style_tags']}

{current_ui_text['speed_optimization']}
{current_ui_text['speed_tips']}

{current_ui_text['tips']}
{current_ui_text['tip_points']}
                """
            )

    generate_btn.click(
        fn=generate_music,
        inputs=[
            lyrics_input,
            tags_input,
            max_duration,
            temperature,
            topk,
            cfg_scale,
        ],
        outputs=[audio_output, download_button],
    )

    gr.Markdown(
        f"""
        --- 
        {current_ui_text['model']} [HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B) |
        {current_ui_text['paper']} [arXiv](https://arxiv.org/abs/2601.10547) |
        {current_ui_text['code']} [GitHub](https://github.com/HeartMuLa/heartlib)

        {current_ui_text['license']}
        """
    )

if __name__ == "__main__":
    demo.launch()