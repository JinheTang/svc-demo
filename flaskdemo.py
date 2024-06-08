# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file
import os
import logging
import soundfile
from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map
from urllib.parse import quote as url_quote

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

app = Flask(__name__)
UPLOAD_FOLDER = 'raw'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def run_inference(model_path="logs/44k/G_37600.pth", config_path="logs/44k/config.json", clip=0, clean_names=["demo.wav"], trans=[0], spk_list=['Kanye'], f0p="crepe", auto_predict_f0=False, \
                cluster_model_path="", cluster_infer_ratio=0, lg=0, enhance=False, \
                shallow_diffusion=False, use_spk_mix=False, loudness_envelope_adjustment=1, fr=False, \
                diffusion_model_path="logs/44k/diffusion/model_0.pt", diffusion_config_path="logs/44k/diffusion/config.yaml", \
                k_step=100, second_encoding=False, only_diffusion=False, slice_db=-40, device=None, noice_scale=0.4, \
                pad_seconds=0.5, wav_format='wav', lgr=0.75, enhancer_adaptive_key=0, cr_threshold=0.05):

    svc_model = Svc(model_path, config_path, None, cluster_model_path, enhance, diffusion_model_path,
                    diffusion_config_path, shallow_diffusion, only_diffusion, use_spk_mix, False)
    
    if len(spk_mix_map) <= 1:
        use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]
    
    infer_tool.fill_a_to_b(trans, clean_names)
    result_paths = []
    for clean_name, tran in zip(clean_names, trans):
        print(f"cleanname: {clean_name}")
        if clean_name == 0:
            break
        raw_audio_path = os.path.join(UPLOAD_FOLDER, clean_name)
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step": k_step,
                "use_spk_mix": use_spk_mix,
                "second_encoding": second_encoding,
                "loudness_envelope_adjustment": loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion:
                isdiffusion = "sovdiff"
            if only_diffusion:
                isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            res_path = os.path.join(RESULT_FOLDER, f'{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}')
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            result_paths.append(res_path)
    
    return result_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_path = request.form['model_path']
        config_path = request.form['config_path']
        # clip = float(request.form['clip'])
        clip = 0
        clean_names = []
        trans = list(map(int, request.form.getlist('trans')))
        spk_list = request.form.getlist('spk_list')
        # auto_predict_f0 = 'auto_predict_f0' in request.form
        # cluster_model_path = request.form['cluster_model_path']
        # cluster_infer_ratio = float(request.form['cluster_infer_ratio'])
        # slice_db = int(request.form['slice_db'])
        # noice_scale = float(request.form['noice_scale'])
        # pad_seconds = float(request.form['pad_seconds'])
        # lg = float(request.form['linear_gradient'])
        # lgr = float(request.form['linear_gradient_retain'])
        f0p = request.form['f0_predictor']
        # enhance = 'enhance' in request.form
        # enhancer_adaptive_key = int(request.form['enhancer_adaptive_key'])
        # cr_threshold = float(request.form['f0_filter_threshold'])
        # diffusion_model_path = request.form['diffusion_model_path']
        # diffusion_config_path = request.form['diffusion_config_path']
        # k_step = int(request.form['k_step'])
        # only_diffusion = 'only_diffusion' in request.form
        # shallow_diffusion = 'shallow_diffusion' in request.form
        # use_spk_mix = 'use_spk_mix' in request.form
        # second_encoding = 'second_encoding' in request.form
        # loudness_envelope_adjustment = float(request.form['loudness_envelope_adjustment'])
        # wav_format = request.form['wav_format']
        # wav_format = 'wav'

        files = request.files.getlist('files')
        print(files)
        for file in files:
            if file:
                filename = file.filename
                if filename.endswith('wav'):
                    print(filename)
                    clean_names.append(filename)
                    print(clean_names, "clean_names")
                file.save(os.path.join(UPLOAD_FOLDER, filename))
        
        result_paths = run_inference(model_path, config_path, clean_names, trans, spk_list, clip,  f0p)
        
        return render_template('results.html', result_paths=result_paths)
    
    return render_template('index.html')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
