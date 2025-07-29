import os
import shutil

def download_omniparser_models():
    from huggingface_hub import hf_hub_download
    weights_path = os.path.join(os.path.dirname(__file__), 'OmniParser', 'weights')
    if not os.path.isdir(os.path.join(weights_path, 'icon_caption_florence')):
        for f1 in ['train_args.yaml', 'model.pt', 'model.yaml', 'LICENSE']:
            hf_hub_download(repo_id='microsoft/OmniParser-v2.0', filename=f'icon_detect/{f1}', local_dir=weights_path)
        for f2 in ['config.json', 'generation_config.json', 'model.safetensors', 'LICENSE']:
            hf_hub_download(repo_id='microsoft/OmniParser-v2.0', filename=f'icon_caption/{f2}', local_dir=weights_path)
        shutil.move(os.path.join(weights_path, 'icon_caption'), os.path.join(weights_path, 'icon_caption_florence'))

def download_paddle_ocr_models():
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(
        lang=os.environ['OCR_LANG'] if 'OCR_LANG' in os.environ else 'japan', #'en',
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,
        det_db_score_mode='slow',
        rec_batch_num=1024)

if __name__ == "__main__":
    download_omniparser_models()
    download_paddle_ocr_models()