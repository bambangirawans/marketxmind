from flask import Blueprint, request, jsonify, render_template, redirect, url_for,  flash, current_app
from app import db
from app.models.customer import Customer
from app.models.invoice import Invoice
from app.models.payment import Payment
from app.models.loyalty import LoyaltyProgram, LoyaltyTier, CustomerLoyalty, ReferralProgram
from app.models.campaign import Campaign, CampaignMetric
from app.models.feedback import CustomerFeedback
from app.forms.advancecrm import LoyaltyProgramForm, CampaignForm, FeedbackForm
from app.utilities.utils import send_whatsapp_message, send_whatsapp_image, send_whatsapp_pdf
from app.utilities.utils import format_phone_number
from datetime import datetime, timedelta
from config import Config
import os
from math import ceil
from sqlalchemy import func, case, and_
from sqlalchemy.exc import IntegrityError,DataError
from sqlalchemy.sql import extract
from PIL import Image
import base64
import logging
import matplotlib.pyplot as plt
import io
import plotly.graph_objs as go
import pandas as pd
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import torch
from flask import Blueprint

advancecrm = Blueprint('advancecrm', __name__)

MODEL_NAME = Config.OPENAI_MODEL
API_KEY = Config.OPENAI_API_KEY

client = OpenAI(
    api_key=API_KEY,
)

model_path = r"/home/ubuntu/marketxmind/Llama-3_2_1B_Instruct/"

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16  # Adjust to torch.float16 if needed
    )

# Tie the model weights
model.tie_weights()

# Load the model with offloading and device mapping
model = load_checkpoint_and_dispatch(
    model,
    model_path,
    device_map="auto",  
    offload_folder=r"/home/ubuntu/marketxmind/offload/"  
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id explicitly
)

# Define conversation structure
conversation = [
    {"role": "system", "content": "Hi, I am a market-X-mind chatbot who is a business analyst specializing in customer loyalty and engagement strategies."},
]

@advancecrm.route('/advancecrm/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Ensure the request contains JSON data
        if request.content_type != 'application/json':
            return jsonify({"response": "error: Content-Type must be 'application/json'"}), 415
        
        user_input = request.get_json().get('message', '').strip()
        if not user_input:
            return jsonify({"response": "error: Invalid input"}), 400
        
        conversation.append({"role": "user", "content": user_input})

        try:
            # Prepare the prompt
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
            # Optionally truncate prompt_text if too long
            max_length = 1024  # or your model's max length
            if len(prompt_text) > max_length:
                prompt_text = prompt_text[-max_length:]

            response = pipe(prompt_text, max_new_tokens=200)

            # Debug the response
            print(response)  # or use logging
            if response and 'text-generation' in response[0]:
                assistant_response = response[0]['text-generation'].strip()
                conversation.append({"role": "assistant", "content": assistant_response})
                return jsonify({"response": assistant_response})
            else:
                return jsonify({"response": "error: no text-generation in response"}), 500

        except Exception as e:
            app.logger.exception("Exception during response generation.")
            return jsonify({"response": f"error: {str(e)}"}), 500
    else:
        return render_template('advancecrm/chat.html', conversation=conversation)


def inital_user_prompt():
   
    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar()
    if not program_total :
        program_total=0
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar()

    if not tier_total :
        tier_total=0
        
    customer_total = db.session.query(func.count(Customer.id)).scalar()
    campaign_total = db.session.query(func.count(Campaign.id)).filter(
        Campaign.is_active == True
    ).scalar()
     
    current_month = func.extract('month', Payment.created_date)
    current_year = func.extract('year', Payment.created_date)
    monthly_revenue = db.session.query(func.sum(Payment.amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    ).scalar()
    
    current_month = func.extract('month', Invoice.created_date)
    current_year = func.extract('year', Invoice.created_date)
    monthly_omzet = db.session.query(func.sum(Invoice.net_amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    ).scalar()

    churn_risk = calculate_churn_risk()  

    system_prompt = f"""
    You are a business analyst specializing in customer loyalty and engagement strategies. You have access to the following data:
    Customer total: {customer_total}
        - Monthly revenue: {monthly_revenue}
        - Monthly omzet: {monthly_omzet}
        - Churn Risk: {churn_risk}%
    Based on this data, please perform the following tasks:
    1.Analyze the current state of customer loyalty and engagement.
    2. Identify key challenges or areas of concern that may impact customer retention.
    3. Provide a list of actionable recommendations to improve customer loyalty programs and enhance customer engagement.
    """

    return system_prompt

def detailed_promp(point_id):
   
    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar() or 0
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar() or 0
    customer_total = db.session.query(func.count(Customer.id)).scalar()
    
    churn_risk = calculate_churn_risk()  
  
    detailed_prompt = f"""
    Deeper and more detailed analysis to implement an increase in points {point_id} at company {current_user.company.name} 
    with number of customers {customer_total} and churn Risk: {churn_risk}%
    Also provide examples and tactical recommendations and metrics that can be obtained within a certain time period.
    Return only the code in your reply and should be represented as a visually appealing HTML div.
    Following structure and for each point and display icon fa-solid that is relevant to content each point for each header :
    <h6><a href="javascript:void(0);"  onclick="getDetailedPoint({{ key_point }})">{{ key_point }}</a></h6>
	<p>{{ detail_point }}</p>
	</div>
    """
    return detailed_prompt

      
@advancecrm.route('/advancecrm')
def dashboard():

    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar()

    if not program_total :
        program_total=0
    
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar()
    if not tier_total :
        tier_total=0
        
    customer_total = db.session.query(func.count(Customer.id)).scalar()
    campaign_total = db.session.query(func.count(Campaign.id)).filter(
        Campaign.is_active == True
    ).scalar()
    
    current_month = func.extract('month', Payment.created_date)
    current_year = func.extract('year', Payment.created_date)
    monthly_revenue = db.session.query(func.sum(Payment.amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    ).scalar()
    
    current_month = func.extract('month', Invoice.created_date)
    current_year = func.extract('year', Invoice.created_date)
    monthly_omzet = db.session.query(func.sum(Invoice.net_amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    ).scalar()
    
    churn_risk = calculate_churn_risk()  
    
           
    return render_template('advancecrm/dashboard.html', 
                            customer_total=customer_total, 
                            monthly_revenue=monthly_revenue,
                            monthly_omzet =monthly_omzet,
                            churn_risk=churn_risk,
                            program_total=program_total,
                            tier_total=tier_total,
                            campaign_total=campaign_total
                            )
                           
def calculate_churn_risk():
    today = datetime.today().date()
    start_of_period = today.replace(day=1) - timedelta(days=1)
    start_of_period = start_of_period.replace(day=1) 

    subquery = (
        db.session.query(
            Invoice.customer_id,
            func.max(Invoice.created_date).label('last_purchase_date')
        )
        .group_by(Invoice.customer_id)
        .subquery()
    )

    result = db.session.query(
        func.count(Customer.id),  # Total customers
        func.count(
            func.nullif(subquery.c.last_purchase_date > start_of_period, False)
        )  
    ).outerjoin(subquery, Customer.id == subquery.c.customer_id).first()

    total_customers, retained_customers = result

    if total_customers == 0:
        return 0 

    lost_customers = total_customers - retained_customers

    churn_rate = (lost_customers / total_customers) * 100
    return round(churn_rate, 2)
    
@advancecrm.route('/advancecrm/recommendations') 
def generate_recommendations():

    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar() or 0
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar() or 0
    customer_total = db.session.query(func.count(Customer.id)).scalar() or 0
    
    campaign_total = db.session.query(func.count(Campaign.id)).filter(
         Campaign.is_active == True
    ).scalar() or 0
    
    current_month = func.extract('month', Payment.created_date)
    current_year = func.extract('year', Payment.created_date)
    revenue_query = db.session.query(func.sum(Payment.amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    )
    monthly_revenue = revenue_query.scalar() or 0
    
    current_month = func.extract('month', Invoice.created_date)
    current_year = func.extract('year', Invoice.created_date)
    omzet_query = db.session.query(func.sum(Invoice.net_amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    )
    monthly_omzet = omzet_query.scalar() or 0
    churn_risk = calculate_churn_risk()  
    
    try:
        
        prompt = f"""
        Analyze a CRM system for a loyalty program 
        that has data :
        Customer total: {customer_total}
        Current monthly revenue : {monthly_revenue}
        Current monthly omzet : {monthly_omzet}
        Churn Rate: {churn_risk}%
        Provide Analyze and recommendations in bahasa Indonesia for creating targeted marketing campaigns using email, SMS, and WhatsApp. 
        Focus on creating customer loyalty programs, offering referral incentives, and personalizing follow-up interactions to reduce churn
        Suggest how to optimize the campaign notifications, targeted discounts and promotions, taking into account customer activity and purchase behavior
        Your reply should be represented as a visually appealing HTML div with the following structure:
        <div class="row g-2 rtl-flex-d-row-r">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header"><h5 class="mb-0">Rekomendasi AI</h5></div>
                        <div class="card-body"></div>
                    </div>
                </div>
        </div>>
        Design Specifications:  professional, modern layout with hover effects to enhance interactivity and easy readability and engagement.
        Return only the code in your reply.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        recommendation = completion.choices[0].message.content

    except Exception as e:
        recommendation= ""
    recommendation = recommendation.replace("`", "").replace("html", "")
    return render_template('advancecrm/rekomendasi.html', 
                            recommendation=recommendation)

@advancecrm.route('/advancecrm/advisor_open_ai') 
def advisor_open_ai():

    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar() or 0
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar() or 0
    customer_total = db.session.query(func.count(Customer.id)).scalar() or 0
    
    campaign_total = db.session.query(func.count(Campaign.id)).filter(
         Campaign.is_active == True
    ).scalar() or 0
    
    current_month = func.extract('month', Payment.created_date)
    current_year = func.extract('year', Payment.created_date)
    revenue_query = db.session.query(func.sum(Payment.amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    )
    monthly_revenue = revenue_query.scalar() or 0
    
    current_month = func.extract('month', Invoice.created_date)
    current_year = func.extract('year', Invoice.created_date)
    omzet_query = db.session.query(func.sum(Invoice.net_amount)).filter(
        current_month == extract('month', func.now()),
        current_year == extract('year', func.now())
    )
    monthly_omzet = omzet_query.scalar() or 0
    churn_risk = calculate_churn_risk()    

    initial_prompt = f"""
    Analyze a CRM system with data:
    - Customer total: {customer_total}
    - Monthly revenue: {monthly_revenue}
    - Monthly omzet: {monthly_omzet}
    - Churn Risk: {churn_risk}%
    Provide analysis and recommendations to improve loyalty programs and customer engagement.
    Provide analysis and recommendations in list of key point with detail explain for each key point 
    Return only the code in your reply without explain and should be represented as a visually appealing HTML card.
    Display Data and Then Display analysis and recommendations Following structure for each key point  and display icon fa-solid that relavance with content each point for each card header :
    <div class="card">
		<div class="card-header"><h5 class="mb-0"><a href="javascript:void(0);"  onclick="getDetailedPoint({{ key_point }})">{{ key_point }}</a></h5></div>
		<div class="card-body"><p>{{ detail_point }}</p></div>
	</div>
                           
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": initial_prompt,
            }],
            max_tokens=5000,
        )

        recommendation = completion.choices[0].message.content
        recommendation = recommendation.replace("`", "").replace("html", "")
    except Exception as e:
        recommendation = "<p>Tidak dapat menghasilkan rekomendasi saat ini. Silakan coba lagi nanti.</p>"
    
    return render_template('advancecrm/advisor.html', recommendations=recommendation)

@advancecrm.route('/advancecrm/advisor_open_ai/detailed-point/<string:point_id>')
def advisor_open_ai_detailed_point(point_id):
   
    program_total = db.session.query(func.count(LoyaltyProgram.id)).scalar() or 0
    tier_total = db.session.query(func.count(LoyaltyTier.id)).scalar() or 0
    customer_total = db.session.query(func.count(Customer.id)).scalar()
    '''
    # Campaigns
    campaign_total = db.session.query(func.count(Campaign.id)).filter_by(company_id=company_id, is_active=True).scalar()

    # Monthly revenue and transaction counts
    current_month = func.extract('month', func.now())
    current_year = func.extract('year', func.now())

    monthly_revenue = db.session.query(func.count(TransactionCrm.id)).filter(
        TransactionCrm.company_id == company_id,
        func.extract('month', TransactionCrm.date) == current_month,
        func.extract('year', TransactionCrm.date) == current_year,
        TransactionCrm.is_rewarded == True
    ).scalar()

    monthly_omzet = db.session.query(func.count(TransactionCrm.id)).filter(
        TransactionCrm.company_id == company_id,
        func.extract('month', TransactionCrm.date) == current_month,
        func.extract('year', TransactionCrm.date) == current_year
    ).scalar()
    '''
    churn_risk = calculate_churn_risk()  
  
    detailed_prompt = f"""
    Conduct a deeper and more detailed analysis to implement improvement point {point_id} for the company 
    with a customer total of {customer_total} and Churn Risk: {churn_risk}%
    Also provide examples and tactical recommendations, along with metrics that can be obtained within a specific timeframe.
    Return only the code  without explain in your reply and represent it as a visually appealing HTML div.
    Follow the structure for each point and display a relevant fa-solid icon for each point's header:
    <h6><a href="javascript:void(0);" onclick="getDetailedPoint({{ key_point }})">{{ key_point }}</a></h6>
    <p>{{ detail_point }}</p>
	</div>
    """
    try:
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
            "role": "user",
            "content": detailed_prompt,
            }],
            max_tokens=5000,
        )

        detailed_response = completion.choices[0].message.content
        detailed_response = detailed_response.replace("`", "").replace("html", "").replace("*", "").replace("#", "")
    except Exception as e:
        detailed_response = "<p>Unable to generate recommendation details at this time. Please try again later.</p>"

    return jsonify(detailed_response=detailed_response)
  


 
  