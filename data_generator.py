"""
Securities Market Fraud Detection System - Data Generator
SEBI Safe Space Initiative

This module generates sample datasets for training and testing the fraud detection system.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

def generate_sample_social_media_data(num_posts=1000):
    """Generate sample social media posts for fraud detection training"""
    
    # Legitimate investment posts
    legitimate_posts = [
        "Consider diversifying your portfolio with index funds for long-term growth",
        "Research shows that systematic investment plans help build wealth over time",
        "Always read the offer document before investing in any mutual fund scheme",
        "Market volatility is normal - stick to your investment plan and don't panic",
        "Consult a SEBI registered advisor before making major investment decisions",
        "Asset allocation is key to managing investment risk effectively",
        "Dollar-cost averaging can help reduce the impact of market volatility",
        "Emergency fund should be your first priority before investing",
        "Tax-saving investments under 80C can provide good returns with tax benefits",
        "Regular portfolio review with rebalancing is essential for long-term success"
    ]
    
    # Fraudulent posts for detection training
    fraudulent_posts = [
        "🚀 GUARANTEED 500% returns in 30 days! Join our premium Telegram group NOW!",
        "BREAKING: XYZ stock will moon tomorrow! Buy before 9 AM. Secret insider info!",
        "💰 I made ₹50 lakhs in one week using this simple trick. DM for details!",
        "URGENT: This penny stock will be the next TESLA! Limited time offer - invest ₹10K get ₹1L",
        "Exclusive IPO allocation guaranteed! No lottery system. Pay only ₹5000 advance",
        "AI trading bot with 99.9% success rate! Make money while you sleep! 🤖💸",
        "Celebrity XYZ recommends this stock in viral video! Don't miss out!",
        "Government approved scheme! Tax-free returns of 40% annually guaranteed!",
        "PUMP ALERT: Everyone buy ABC stock at 2 PM sharp! Let's take it to the moon! 🌙",
        "SECRET FORMULA REVEALED! Turn ₹1000 into ₹1 lakh in 15 days! Limited seats!"
    ]
    
    posts_data = []
    
    for i in range(num_posts):
        if random.random() < 0.3:  # 30% fraudulent posts
            post_text = random.choice(fraudulent_posts)
            label = "fraudulent"
            engagement = random.randint(500, 5000)  # Higher engagement for fraud
            verified_user = False
        else:
            post_text = random.choice(legitimate_posts)
            label = "legitimate"
            engagement = random.randint(10, 500)
            verified_user = random.choice([True, False])
        
        posts_data.append({
            'post_id': f'post_{i:04d}',
            'text': post_text,
            'platform': random.choice(['telegram', 'whatsapp', 'twitter', 'facebook', 'instagram']),
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
            'engagement': engagement,
            'label': label,
            'user_verified': verified_user,
            'follower_count': random.randint(100, 50000),
            'account_age_days': random.randint(1, 1000)
        })
    
    return pd.DataFrame(posts_data)

def generate_advisor_database():
    """Generate sample advisor database with legitimate and fraudulent advisors"""
    
    legitimate_advisors = [
        {"name": "HDFC Securities", "license": "INZ000183735", "status": "Active", "sebi_registered": True, "registration_date": "2010-05-15"},
        {"name": "ICICI Direct", "license": "INZ000183631", "status": "Active", "sebi_registered": True, "registration_date": "2009-03-20"},
        {"name": "Zerodha", "license": "INZ000031633", "status": "Active", "sebi_registered": True, "registration_date": "2010-08-15"},
        {"name": "Angel Broking", "license": "INZ000156038", "status": "Active", "sebi_registered": True, "registration_date": "2008-12-10"},
        {"name": "Upstox", "license": "INZ000156532", "status": "Active", "sebi_registered": True, "registration_date": "2009-01-25"},
        {"name": "Groww", "license": "INZ000301838", "status": "Active", "sebi_registered": True, "registration_date": "2020-06-12"},
        {"name": "5paisa", "license": "INZ000156038", "status": "Active", "sebi_registered": True, "registration_date": "2016-11-03"}
    ]
    
    fraudulent_advisors = [
        {"name": "QuickRich Advisors", "license": "FAKE001", "status": "Fraud", "sebi_registered": False, "registration_date": None},
        {"name": "FastProfit Securities", "license": "FAKE002", "status": "Fraud", "sebi_registered": False, "registration_date": None},
        {"name": "GoldMine Investments", "license": "FAKE003", "status": "Fraud", "sebi_registered": False, "registration_date": None},
        {"name": "InstantWealth Partners", "license": "FAKE004", "status": "Fraud", "sebi_registered": False, "registration_date": None},
        {"name": "MegaReturn Consultancy", "license": "FAKE005", "status": "Fraud", "sebi_registered": False, "registration_date": None}
    ]
    
    return {"legitimate": legitimate_advisors, "fraudulent": fraudulent_advisors}

def generate_corporate_announcements(num_announcements=500):
    """Generate sample corporate announcements"""
    
    companies = [
        'TCS', 'Infosys', 'HDFC Bank', 'Reliance Industries', 'ITC', 'State Bank of India',
        'Wipro', 'HUL', 'Bharti Airtel', 'HDFC', 'ICICI Bank', 'Larsen & Toubro',
        'Asian Paints', 'Maruti Suzuki', 'Kotak Mahindra Bank', 'Titan Company'
    ]
    
    legitimate_announcements = [
        "Board meeting scheduled to consider quarterly results on {date}",
        "Dividend of ₹{amount} per share declared for shareholders",
        "Company signs MoU with {partner} for strategic collaboration",
        "Q{quarter} results show {growth}% growth in revenue",
        "Board approves bonus issue in ratio 1:1",
        "Rights issue approved by board at ₹{price} per share",
        "Company launches new product line for {segment} market",
        "Annual general meeting scheduled for {date}",
        "Board recommends final dividend of ₹{amount} per share",
        "Company receives regulatory approval for {project} project"
    ]
    
    fraudulent_announcements = [
        "BREAKING: Company acquired by Apple for $50 billion deal!",
        "EXCLUSIVE: Government grants ₹10,000 crore subsidy to company",
        "URGENT: Stock split announced in ratio 10:1 effective immediately",
        "Company discovers oil reserves worth ₹1 lakh crore in Rajasthan",
        "Merger with Tesla confirmed - stock to increase 1000%",
        "Secret government contract worth ₹50,000 crore finalized",
        "Revolutionary technology breakthrough - patents worth billions",
        "Exclusive mining rights granted in Antarctica",
        "Joint venture with SpaceX for Mars mission announced",
        "Cryptocurrency partnership with major bank confirmed"
    ]
    
    announcements_data = []
    
    for i in range(num_announcements):
        if random.random() < 0.15:  # 15% fraudulent announcements
            template = random.choice(fraudulent_announcements)
            label = "fraudulent"
            credibility_score = random.uniform(0.1, 0.4)
            source_reliability = "Low"
        else:
            template = random.choice(legitimate_announcements)
            label = "legitimate"
            credibility_score = random.uniform(0.7, 0.95)
            source_reliability = "High"
        
        company = random.choice(companies)
        announcement_text = template.format(
            date=(datetime.now() + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            amount=random.randint(5, 50),
            partner=random.choice(['Microsoft', 'Google', 'Amazon', 'IBM', 'Oracle']),
            quarter=random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
            growth=random.randint(5, 25),
            price=random.randint(100, 1000),
            segment=random.choice(['automobile', 'healthcare', 'technology', 'retail']),
            project=random.choice(['expansion', 'manufacturing', 'research', 'infrastructure'])
        )
        
        announcements_data.append({
            'announcement_id': f'ann_{i:04d}',
            'company': company,
            'text': announcement_text,
            'timestamp': datetime.now() - timedelta(days=random.randint(0, 90)),
            'source': random.choice(['BSE', 'NSE', 'Company Website', 'PR Agency', 'News Portal']),
            'label': label,
            'credibility_score': credibility_score,
            'market_impact': random.choice(['High', 'Medium', 'Low']),
            'source_reliability': source_reliability,
            'verification_status': 'Verified' if label == 'legitimate' else 'Unverified'
        })
    
    return pd.DataFrame(announcements_data)

def generate_market_data():
    """Generate sample market data for analysis"""
    
    stocks = ['TCS', 'INFY', 'HDFCBANK', 'RELIANCE', 'ITC', 'SBIN', 'WIPRO', 'BHARTIARTL']
    market_data = []
    
    for stock in stocks:
        base_price = random.uniform(100, 2000)
        base_volume = random.uniform(1000000, 10000000)
        
        for i in range(90):  # 90 days of data
            date = datetime.now() - timedelta(days=i)
            
            # Simulate normal price movement
            price_change = random.uniform(-0.05, 0.05)  # ±5% daily change
            new_price = base_price * (1 + price_change)
            
            # Volume correlation with price change
            volume_multiplier = 1 + (abs(price_change) * 3)
            volume = int(base_volume * volume_multiplier)
            
            market_data.append({
                'stock': stock,
                'date': date,
                'open_price': base_price,
                'close_price': new_price,
                'high_price': max(base_price, new_price) * 1.02,
                'low_price': min(base_price, new_price) * 0.98,
                'volume': volume,
                'price_change_percent': price_change * 100
            })
            
            base_price = new_price
    
    return pd.DataFrame(market_data)

def main():
    """Main function to generate all sample datasets"""
    
    print("🔄 Generating sample datasets for fraud detection system...")
    
    # Create data directory
    os.makedirs('data/sample_data', exist_ok=True)
    
    # Generate social media data
    print("📱 Generating social media data...")
    social_media_df = generate_sample_social_media_data(2000)
    social_media_df.to_csv('data/sample_data/social_media_posts.csv', index=False)
    
    # Generate advisor database
    print("👥 Generating advisor database...")
    advisor_db = generate_advisor_database()
    with open('data/sample_data/advisor_database.json', 'w') as f:
        json.dump(advisor_db, f, indent=2, default=str)
    
    # Generate corporate announcements
    print("📢 Generating corporate announcements...")
    announcements_df = generate_corporate_announcements(1000)
    announcements_df.to_csv('data/sample_data/corporate_announcements.csv', index=False)
    
    # Generate market data
    print("📈 Generating market data...")
    market_df = generate_market_data()
    market_df.to_csv('data/sample_data/market_data.csv', index=False)
    
    # Generate summary statistics
    summary = {
        'generation_date': datetime.now().isoformat(),
        'datasets': {
            'social_media_posts': {
                'total_posts': len(social_media_df),
                'fraudulent_posts': len(social_media_df[social_media_df['label'] == 'fraudulent']),
                'legitimate_posts': len(social_media_df[social_media_df['label'] == 'legitimate']),
                'platforms': social_media_df['platform'].unique().tolist()
            },
            'corporate_announcements': {
                'total_announcements': len(announcements_df),
                'fraudulent_announcements': len(announcements_df[announcements_df['label'] == 'fraudulent']),
                'legitimate_announcements': len(announcements_df[announcements_df['label'] == 'legitimate']),
                'companies': len(announcements_df['company'].unique())
            },
            'advisor_database': {
                'legitimate_advisors': len(advisor_db['legitimate']),
                'fraudulent_advisors': len(advisor_db['fraudulent'])
            },
            'market_data': {
                'stocks': len(market_df['stock'].unique()),
                'total_records': len(market_df),
                'date_range': f"{market_df['date'].min()} to {market_df['date'].max()}"
            }
        }
    }
    
    with open('data/sample_data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("✅ Sample datasets generated successfully!")
    print(f"📊 Social Media Posts: {len(social_media_df)} ({len(social_media_df[social_media_df['label'] == 'fraudulent'])} fraudulent)")
    print(f"📊 Corporate Announcements: {len(announcements_df)} ({len(announcements_df[announcements_df['label'] == 'fraudulent'])} fraudulent)")
    print(f"📊 Advisor Database: {len(advisor_db['legitimate']) + len(advisor_db['fraudulent'])} total advisors")
    print(f"📊 Market Data: {len(market_df)} records for {len(market_df['stock'].unique())} stocks")
    print("📁 All files saved to 'data/sample_data/' directory")

if __name__ == "__main__":
    main()