import sys
import os.path as osp
import folder_paths
now_dir = osp.dirname(__file__)
sys.path.append(now_dir)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
glm_dir = osp.join(aifsh_dir,"GLM4Voice")
glm4voice_9b = osp.join(glm_dir,"glm-4-voice-9b")
glm4voice_9b_int8 = osp.join(glm_dir,"glm-4-voice-9b-int8")
glm4voice_9b_int4 = osp.join(glm_dir,"glm-4-voice-9b-int4")
glm4voice_9b_decoder = osp.join(glm_dir,"glm-4-voice-decoder")
glm4voice_9b_tokener = osp.join(glm_dir,"glm-4-voice-tokenizer")

import uuid
import torch
import json
import tempfile
import torchaudio
from comfy.utils import ProgressBar
from huggingface_hub import snapshot_download
from flow_inference import AudioDecoder
from model_server import ModelWorker
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
history = []
previous_input_tokens=""
previous_completion_tokens=""
class GLM4VoiceNode:
    def __init__(self):
        if not osp.exists(osp.join(glm4voice_9b_decoder,"flow.pt")):
            snapshot_download(repo_id="THUDM/glm-4-voice-decoder",local_dir=glm4voice_9b_decoder)
        '''
        if not osp.exists(osp.join(glm4voice_9b_tokener,"model.safetensors")):
            snapshot_download(repo_id="THUDM/glm-4-voice-tokenizer",local_dir=glm4voice_9b_decoder)
        '''
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "text":("TEXT",),
                "temperature":("FLOAT",{
                    "default":0.2,
                    "display":"slider",
                    "setp":0.05,
                    "ronnd":0.001,
                    "max":1,
                    "min":0,
                }),
                "top_p":("FLOAT",{
                    "default":0.8,
                    "display":"slider",
                    "setp":0.05,
                    "ronnd":0.001,
                    "max":1,
                    "min":0,
                }),
                "max_new_token":("INT",{
                    "default":2000,
                }),
                "model_type":(['oringnal',"int8","int4"],),
                "clear_history":("BOOLEAN",{
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO","*","*",)
    RETURN_NAMES = ("current_audio","current_text","history",)

    FUNCTION = "chat"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_GLM4Voice"
    
    def chat(self,text,temperature,top_p,max_new_token,model_type,clear_history):
        
        global history,previous_input_tokens,previous_completion_tokens
        if clear_history:
            history = []
            previous_input_tokens=""
            previous_completion_tokens=""

        if model_type == "oringnal":
            model_path = glm4voice_9b
            if not osp.exists(osp.join(glm4voice_9b,"model-00004-of-00004.safetensors")):
                snapshot_download(repo_id="THUDM/glm-4-voice-9b",
                                local_dir=glm4voice_9b)
            glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                   device=device).eval()
        elif model_type == "int8":
            model_path = glm4voice_9b_int8
            if not osp.exists(osp.join(model_path,"model-00003-of-00003.safetensors")):
                snapshot_download(repo_id="cydxg/glm-4-voice-9b-int8",
                                local_dir=glm_dir,
                                allow_patterns=["glm-4-voice-9b-int8/*"])
            glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                   device_map=device,
                                                   low_cpu_mem_usage=True,load_in_8bit=True).eval()
        else:
            model_path = glm4voice_9b_int4
            if not osp.exists(osp.join(model_path,"model-00002-of-00002.safetensors")):
                snapshot_download(repo_id="cydxg/glm-4-voice-9b-int4",
                                local_dir=glm_dir,
                                allow_patterns=["glm-4-voice-9b-int4/*"])
            glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                   device_map=device,
                                                   low_cpu_mem_usage=True,load_in_4bit=True).eval()
        
        ## init
        # GLM
        glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Flow & Hift
        audio_decoder = AudioDecoder(config_path=osp.join(glm4voice_9b_decoder,"config.yaml"),
                                     flow_ckpt_path=osp.join(glm4voice_9b_decoder,"flow.pt"),
                                     hift_ckpt_path=osp.join(glm4voice_9b_decoder,"hift.pt"),
                                     device=device)
        
        user_input = text
        history.append({"role": "user", "content": text})
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()
        if "<|system|>" not in inputs:
            inputs += f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        
        
        with torch.no_grad():
            worker = ModelWorker(device)
            worker.glm_model = glm_model
            worker.glm_tokenizer = glm_tokenizer
            params = {
                "prompt": inputs,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_token,
            }
            text_generater = worker.generate_stream_gate(params)
            text_tokens, audio_tokens = [], []
            audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            end_token_id = glm_tokenizer.convert_tokens_to_ids('<|user|>')
            complete_tokens = []
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
            this_uuid = str(uuid.uuid4())
            tts_speechs = []
            tts_mels = []
            prev_mel = None
            is_finalize = False
            block_size = 10
            for chunk in text_generater:
                token_id = json.loads(chunk)["token_id"]
                if token_id == end_token_id:
                    is_finalize = True
                if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                    block_size = 20
                    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)

                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                  prompt_token=flow_prompt_speech_token.to(device),
                                                                  prompt_feat=prompt_speech_feat.to(device),
                                                                  finalize=is_finalize)
                    prev_mel = tts_mel

                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    # yield history, inputs, '', '', (22050, tts_speech.squeeze().cpu().numpy()), None
                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                    audio_tokens = []
                if not is_finalize:
                    complete_tokens.append(token_id)
                    if token_id >= audio_offset:
                        audio_tokens.append(token_id - audio_offset)
                    else:
                        text_tokens.append(token_id)
        
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        complete_text = glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,dir=folder_paths.get_output_directory()) as f:
            torchaudio.save(f, tts_speech.unsqueeze(0), 22050, format="wav")
        answer_text = glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
        # answer_text = answer_text.decode()
        history.append({"role": "assistant", "content": f"{f.name}\t{answer_text}"})
        previous_input_tokens = inputs
        previous_completion_tokens = complete_text
        res_audio = {
            "waveform": tts_speech.unsqueeze(0).unsqueeze(0),
            "sample_rate":22050
        }
        return (res_audio,answer_text,history,)
        

NODE_CLASS_MAPPINGS = {
    "GLM4VoiceNode": GLM4VoiceNode
}