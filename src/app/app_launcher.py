from app.StreamlitApp import StreamlitApp
from rag.pipeline_factory import create_pipeline
import certifi
import os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()


if __name__ == "__main__":
    pipeline = create_pipeline()
    app = StreamlitApp(pipeline)
    app.run()
