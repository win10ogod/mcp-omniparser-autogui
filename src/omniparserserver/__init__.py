#coding: utf-8

def main():
    import os
    import sys
    from fastapi import FastAPI
    import uvicorn

    omniparser_path = os.path.join(os.path.dirname(__file__), '..', '..', 'OmniParser')
    omniparser_cd_path = os.path.join(omniparser_path, 'omnitool', 'omniparserserver')
    os.chdir(omniparser_cd_path)
    sys.path = [omniparser_path, omniparser_cd_path, os.path.join(os.path.dirname(__file__), '..', '..')] + sys.path
    from download_models import download_omniparser_models
    download_omniparser_models()
    from omnitool.omniparserserver.omniparserserver import args

    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)