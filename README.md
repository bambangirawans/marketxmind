# Setup Instructions

## 1. Install Python and pip
```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

## 2. Install Git
```bash
sudo apt install git -y
```

## 3. Clone Repository
```bash
git clone https://github.com/bambangirawans/marketxmind.git
cd marketxmind
```

## 4. Set Up the Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## 5. Install Required Packages
```bash
pip install -r requirements.txt
```

## 6. Configure `config.py`
* Edit `config.py` with appropriate settings.

## 7. Install and Configure Nginx
```bash
sudo apt install nginx -y
sudo nano /etc/nginx/sites-available/marketxmind
```

```nginx
server {
    listen 80;
    server_name your_domain_or_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/marketxmind /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

## 8. Create Database Using Flask Migration
```bash
flask db init
flask db migrate
flask db upgrade
```

## 9. Install and Configure Gunicorn
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

## 10. Automate Application Startup (Optional)
```bash
sudo nano /etc/systemd/system/marketxmind.service
```

```ini
[Unit]
Description=Gunicorn instance to serve marketxmind
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/path/to/marketxmind
Environment="PATH=/path/to/marketxmind/venv/bin"
ExecStart=/path/to/marketxmind/venv/bin/gunicorn --workers 3 --bind unix:marketxmind.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl start marketxmind
sudo systemctl enable marketxmind
```