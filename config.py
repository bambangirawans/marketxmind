import os
from datetime import timedelta
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.urandom(24).hex()
    SALT_KEY='your-market-X-mind-salt'
    SQLALCHEMY_DATABASE_URI =  'mysql+pymysql://user:password@your-end-point.rds.amazonaws.com/dbmarketxmind' 
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAIL_SERVER = 'mail.your-domain.com'
    MAIL_PORT = 465
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_USERNAME = 'noresponse@your-domain.com'
    MAIL_PASSWORD = 'your-password'
    MAIL_DEFAULT_SENDER = ('MarketXmind', 'noresponse@your-domain.com')
    MAIL_DEFAULT_CC = ('MarketXmind', 'hello@your-domain.com')
    MAIL_DEFAULT_BCC = ('MarketXmind', 'support@your-domain.com')
    DATE_FORMAT = '%d-%m-%Y'
    UPLOAD_FOLDER = 'uploads'
    REMEMBER_COOKIE_DURATION = 60 * 60 * 24 * 1
    PDF_KIT_PATH ='/usr/bin/wkhtmltopdf'
    IMG_KIT_PATH ='/usr/bin/wkhtmltoimage'
    SESSION_TYPE = 'filesystem'   
    SESSION_FILE_DIR = os.path.join(basedir, 'marketxmind_session')  
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    OPENAI_API_KEY="your-open-ai-key"
    OPENAI_MODEL="gpt-4o-mini"
    AIHF_KEY="your-hf-key"
    LLAMA_INSTRUCT_MODEL_LIGHT_NAME="/home/ubuntu/marketxmind/Llama-3_2_1B_Instruct/" 
    LLAMA_VISION_MODEL_NAME="/home/ubuntu/marketxmind/Llama-3_2_11B_Vision_Instruct/"
    LLAMA_INSTRUCT_MODEL_NAME="/home/ubuntu/marketxmind/Llama-3_1_70B_Instruct/"

