"""
SEBI Fraud Detection System - Web Application
Google Careers-inspired Design

A comprehensive web application for fraud detection using advanced AI agents
and multi-modal analysis techniques from the scrapper.ipynb notebook.

Features:
- Content scanning for fraud indicators
- Advisor credential verification  
- Social media monitoring
- Corporate announcement analysis
- Real-time risk assessment
- Google-inspired UI/UX design
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import uvicorn
import os
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# Import fraud detection components from scrapper.ipynb techniques
try:
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.core.workflow import Context
    from google.genai import types
    import nltk
    from textblob import TextBlob
    from sklearn.feature_extraction.text import TfidfVectorizer
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False
    print("⚠️  AI components not available. Running in demo mode.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="SEBI Fraud Detection System",
    description="Comprehensive fraud detection platform inspired by Google's design principles",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API requests/responses
class ContentScanRequest(BaseModel):
    content: str = Field(..., description="Content to scan for fraud indicators")
    content_type: str = Field(default="text", description="Type of content (text, url, document)")
    
class AdvisorVerificationRequest(BaseModel):
    advisor_name: str = Field(..., description="Name of the investment advisor")
    registration_number: Optional[str] = Field(None, description="SEBI registration number")
    
class SocialMediaMonitorRequest(BaseModel):
    platform: str = Field(..., description="Social media platform to monitor")
    search_terms: str = Field(..., description="Terms to search for suspicious content")
    
class CorporateAnnouncementRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    announcement_text: str = Field(..., description="Corporate announcement content")
    
class FraudAnalysisResult(BaseModel):
    risk_score: float = Field(..., description="Risk score from 0-100")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    detected_patterns: List[str] = Field(..., description="List of detected fraud patterns")
    recommendations: List[str] = Field(..., description="Recommended actions")
    timestamp: str = Field(..., description="Analysis timestamp")

# Global state for fraud detection system
fraud_detection_state = {
    "total_scans": 0,
    "high_risk_detections": 0,
    "advisor_verifications": 0,
    "social_media_alerts": 0,
    "system_uptime": datetime.now(),
    "active_threats": []
}

# Initialize AI components if available
llm = None
llm_with_search = None

def initialize_ai_system():
    """Initialize the AI fraud detection system using techniques from scrapper.ipynb"""
    global llm, llm_with_search
    
    if not AI_ENABLED:
        return False
        
    try:
        # Get API key from environment
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("No Gemini API key found. AI features will be limited.")
            return False
            
        # Initialize Gemini 2.5 Flash for fraud detection
        llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
        
        # Enhanced LLM with search capabilities
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        llm_with_search = GoogleGenAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            generation_config=types.GenerateContentConfig(tools=[google_search_tool])
        )
        
        logger.info("✅ AI fraud detection system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing AI system: {e}")
        return False

# Initialize database
def init_database():
    """Initialize SQLite database for fraud detection data"""
    db_path = Path("data/regulatory_db/fraud_detection.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Content scans table with enhanced fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            risk_score REAL,
            risk_level TEXT,
            detected_patterns TEXT,
            recommendations TEXT,
            detailed_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Advisor verifications table with enhanced fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS advisor_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            advisor_name TEXT NOT NULL,
            registration_number TEXT,
            is_legitimate BOOLEAN,
            verification_status TEXT,
            risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Social media monitoring table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS social_media_monitoring (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            search_terms TEXT NOT NULL,
            findings TEXT,
            threat_level TEXT,
            detailed_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Corporate announcements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corporate_announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            announcement_text TEXT NOT NULL,
            credibility_score REAL,
            authenticity_assessment TEXT,
            detailed_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully!")

# Advanced fraud detection functions using techniques from scrapper.ipynb
async def scan_content_for_fraud(content: str, content_type: str = "text") -> Dict:
    """
    Comprehensive content scanning using multi-agent AI analysis from scrapper.ipynb
    Provides detailed fraud indicator analysis with professional assessment
    """
    
    # Enhanced fraud indicators with categorization (from scrapper.ipynb)
    fraud_patterns = {
        "unrealistic_returns": [
            "guaranteed returns", "500% returns", "800% returns", "risk-free", "guaranteed profits",
            "no risk", "100% guaranteed", "assured returns", "fail-proof", "never loses"
        ],
        "urgency_tactics": [
            "limited time offer", "act now", "urgent action required", "miss out forever",
            "limited slots", "24 hours left", "exclusive opportunity", "hurry up", "last chance"
        ],
        "secrecy_claims": [
            "secret algorithm", "insider tips", "proprietary method", "billionaire secrets",
            "exclusive strategy", "hidden formula", "confidential", "insider information"
        ],
        "social_manipulation": [
            "join our group", "whatsapp group", "telegram channel", "exclusive community",
            "delete after reading", "share with friends", "viral opportunity"
        ],
        "fake_credentials": [
            "certified expert", "investment guru", "proven track record", "award winning",
            "government approved", "sebi certified", "rbi approved"
        ],
        "pressure_tactics": [
            "no experience needed", "get rich quick", "instant wealth", "passive income",
            "work from home", "easy money", "effortless profits"
        ]
    }
    
    detected_categories = {}
    total_risk_score = 0
    detailed_findings = []
    
    content_lower = content.lower()
    
    # Analyze each category
    for category, patterns in fraud_patterns.items():
        category_matches = []
        for pattern in patterns:
            if pattern in content_lower:
                category_matches.append(pattern)
                total_risk_score += 15  # Higher scoring for better detection
        
        if category_matches:
            detected_categories[category] = category_matches
    
    # AI-powered comprehensive analysis
    if llm_with_search:
        try:
            comprehensive_prompt = f"""
            🚨 CONDUCT COMPREHENSIVE FRAUD DETECTION ANALYSIS 🚨
            
            As an expert securities fraud analyst, analyze this investment content for SEBI compliance:
            
            CONTENT TO ANALYZE:
            {content}
            
            PROVIDE DETAILED ANALYSIS INCLUDING:
            
            1. **FRAUD INDICATORS ASSESSMENT:**
            - Identify specific unrealistic return promises and explain why they're impossible
            - Analyze pressure tactics and urgency language
            - Evaluate claims about "secret" methods or insider information
            - Check for proper regulatory disclaimers and warnings
            
            2. **ADVISOR CREDENTIAL VERIFICATION:**
            - Verify any mentioned advisor names and registration numbers
            - Check SEBI database for legitimate registration
            - Identify fake or fabricated license numbers
            - Cross-reference with known fraudulent entities
            
            3. **RED FLAGS ANALYSIS:**
            - Unsolicited investment offers
            - Social media group invitations
            - Lack of proper documentation
            - Missing risk disclosures
            
            4. **REGULATORY COMPLIANCE:**
            - SEBI investment advisor regulations compliance
            - Missing mandatory disclosures
            - Violation of advertising norms
            
            5. **CONCLUSION & RECOMMENDATIONS:**
            - Overall fraud risk assessment
            - Specific actions investors should take
            - Reporting mechanisms for suspected fraud
            
            Format your response as a detailed professional fraud analysis report.
            Use specific examples from the content and provide actionable recommendations.
            """
            
            response = await llm_with_search.acomplete(comprehensive_prompt)
            detailed_analysis = str(response)
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            detailed_analysis = "AI analysis temporarily unavailable"
    else:
        # Fallback detailed analysis when AI is unavailable
        detailed_analysis = generate_fallback_analysis(content, detected_categories)
    
    # Calculate final risk score
    risk_score = min(total_risk_score, 100)
    
    # Determine risk level with more granular assessment
    if risk_score >= 80:
        risk_level = "Critical"
    elif risk_score >= 60:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Medium"
    elif risk_score >= 10:
        risk_level = "Low"
    else:
        risk_level = "Minimal"
    
    # Generate comprehensive recommendations
    recommendations = generate_detailed_recommendations(risk_level, detected_categories)
    
    # Store in database with enhanced details
    try:
        db_path = Path("data/regulatory_db/fraud_detection.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO content_scans 
            (content, content_type, risk_score, risk_level, detected_patterns, recommendations)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            content[:1000],  # Store more content for analysis
            content_type,
            risk_score,
            risk_level,
            json.dumps(detected_categories),
            json.dumps(recommendations)
        ))
        
        conn.commit()
        conn.close()
        
        # Update global state
        fraud_detection_state["total_scans"] += 1
        if risk_level in ["High", "Critical"]:
            fraud_detection_state["high_risk_detections"] += 1
            
    except Exception as e:
        logger.error(f"Database error: {e}")
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detected_patterns": list(detected_categories.keys()),
        "detailed_findings": detected_categories,
        "recommendations": recommendations,
        "detailed_analysis": detailed_analysis,
        "timestamp": datetime.now().isoformat()
    }

def generate_fallback_analysis(content: str, detected_categories: Dict) -> str:
    """Generate detailed fallback analysis when AI is unavailable"""
    
    analysis_parts = []
    analysis_parts.append("🚨 FRAUD DETECTION ANALYSIS:")
    analysis_parts.append("=" * 50)
    
    if detected_categories:
        analysis_parts.append("This content displays multiple red flags commonly associated with fraudulent investment schemes.")
        analysis_parts.append("\n**FRAUD INDICATORS DETECTED:**\n")
        
        if "unrealistic_returns" in detected_categories:
            analysis_parts.append("• **Unrealistic Returns**: The content promises extraordinary returns that are not achievable through legitimate investments. Legitimate investments always carry risk, and no financial professional can guarantee high returns.")
            
        if "urgency_tactics" in detected_categories:
            analysis_parts.append("• **Pressure Tactics**: The use of urgency language creates artificial time pressure to prevent proper due diligence. This is a classic manipulation technique used by fraudsters.")
            
        if "secrecy_claims" in detected_categories:
            analysis_parts.append("• **Secret Methods**: Claims about proprietary algorithms or insider information are used to obscure the lack of legitimate investment strategy. Legitimate advisors can explain their methods clearly.")
            
        if "social_manipulation" in detected_categories:
            analysis_parts.append("• **Social Media Tactics**: Promotion through private groups or messaging apps is a common channel for fraudulent schemes, avoiding regulatory oversight.")
            
        if "fake_credentials" in detected_categories:
            analysis_parts.append("• **Credential Issues**: Any credentials mentioned should be independently verified with SEBI before proceeding.")
    
    else:
        analysis_parts.append("The content appears to have minimal fraud indicators based on initial pattern analysis.")
    
    analysis_parts.append("\n**RECOMMENDATIONS:**")
    analysis_parts.append("• Always verify advisor credentials with SEBI directly")
    analysis_parts.append("• Be skeptical of guaranteed returns or get-rich-quick schemes") 
    analysis_parts.append("• Conduct thorough due diligence before investing")
    analysis_parts.append("• Report suspicious activities to authorities")
    
    analysis_parts.append("=" * 50)
    
    return "\n".join(analysis_parts)

def generate_detailed_recommendations(risk_level: str, detected_categories: Dict) -> List[str]:
    """Generate detailed, actionable recommendations based on risk assessment"""
    
    recommendations = []
    
    if risk_level in ["Critical", "High"]:
        recommendations.extend([
            "🚨 CRITICAL FRAUD ALERT: Do not engage with this investment offer under any circumstances",
            "🔍 The content shows multiple sophisticated fraud indicators - this is likely a coordinated scam",
            "📱 Report this content immediately to SEBI, local police cyber crime unit, and the platform hosting it",
            "⚠️ If you have already shared personal information, monitor your accounts for unauthorized activity",
            "🛡️ Warn friends and family about this specific fraud pattern to prevent further victimization"
        ])
        
        if "fake_credentials" in detected_categories:
            recommendations.append("📋 The mentioned advisor credentials appear fraudulent - verify any financial advisor at https://www.sebi.gov.in")
            
    elif risk_level == "Medium":
        recommendations.extend([
            "⚡ Exercise extreme caution - this content shows concerning fraud indicators",
            "🔍 Thoroughly research any mentioned advisors through official SEBI channels",
            "📋 Demand proper documentation, regulatory disclosures, and risk warnings",
            "💼 Consult with a verified, SEBI-registered financial advisor before proceeding",
            "📱 Consider reporting suspicious elements to relevant authorities"
        ])
        
    elif risk_level == "Low":
        recommendations.extend([
            "⚠️ Some potentially concerning elements detected - proceed with standard due diligence",
            "📋 Verify all claims and credentials independently",
            "💼 Ensure proper documentation and regulatory compliance",
            "🔍 Cross-check investment details with official sources"
        ])
        
    else:
        recommendations.extend([
            "✅ Content appears relatively safe based on fraud pattern analysis",
            "📋 Still recommended to verify advisor credentials and investment details",
            "💼 Follow standard investment due diligence practices"
        ])
    
    return recommendations

async def verify_advisor_credentials(advisor_name: str, registration_number: str = None) -> Dict:
    """
    Comprehensive SEBI advisor verification using database lookups and AI analysis
    Provides detailed credential validation with professional assessment
    """
    
    verification_results = {
        "is_legitimate": False,
        "verification_status": "Unknown",
        "detailed_analysis": "",
        "credentials_summary": {},
        "risk_assessment": {},
        "recommendations": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Load SEBI advisor database
    try:
        db_path = Path("data/regulatory_db/sebi_advisors.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Search by name and/or registration number
            query_conditions = []
            query_params = []
            
            if advisor_name:
                query_conditions.append("LOWER(name) LIKE ?")
                query_params.append(f"%{advisor_name.lower()}%")
                
            if registration_number:
                query_conditions.append("registration_number = ?")
                query_params.append(registration_number)
            
            if query_conditions:
                query = f"SELECT * FROM advisors WHERE {' AND '.join(query_conditions)}"
                cursor.execute(query, query_params)
                advisor_records = cursor.fetchall()
                
                if advisor_records:
                    # Found matching advisor(s)
                    advisor_data = advisor_records[0]  # Take first match
                    verification_results["is_legitimate"] = True
                    verification_results["verification_status"] = "VERIFIED"
                    
                    verification_results["credentials_summary"] = {
                        "name": advisor_data[1],
                        "registration_number": advisor_data[2], 
                        "registration_date": advisor_data[3],
                        "status": advisor_data[4],
                        "address": advisor_data[5],
                        "phone": advisor_data[6],
                        "email": advisor_data[7]
                    }
                    
                    # Additional verification checks
                    status = advisor_data[4]
                    if status and "suspended" in status.lower():
                        verification_results["is_legitimate"] = False
                        verification_results["verification_status"] = "SUSPENDED"
                        verification_results["risk_assessment"] = {
                            "risk_level": "Critical",
                            "reason": "Advisor registration has been suspended by SEBI"
                        }
                    elif status and "cancelled" in status.lower():
                        verification_results["is_legitimate"] = False
                        verification_results["verification_status"] = "CANCELLED"
                        verification_results["risk_assessment"] = {
                            "risk_level": "Critical",
                            "reason": "Advisor registration has been cancelled by SEBI"
                        }
                    else:
                        verification_results["risk_assessment"] = {
                            "risk_level": "Low",
                            "reason": "Advisor appears to be legitimately registered with SEBI"
                        }
                else:
                    # No matching advisor found
                    verification_results["verification_status"] = "NOT_FOUND"
                    verification_results["risk_assessment"] = {
                        "risk_level": "Critical",
                        "reason": "Advisor not found in SEBI database - likely fraudulent"
                    }
                    
            conn.close()
            
    except Exception as e:
        logger.error(f"Database verification error: {e}")
        verification_results["verification_status"] = "DATABASE_ERROR"
        verification_results["risk_assessment"] = {
            "risk_level": "High",
            "reason": "Unable to verify credentials due to database access issue"
        }
    
    # AI-powered comprehensive analysis
    if llm_with_search:
        try:
            verification_prompt = f"""
            🔍 CONDUCT COMPREHENSIVE SEBI ADVISOR VERIFICATION 🔍
            
            As a securities regulatory expert, analyze these advisor credentials:
            
            ADVISOR DETAILS:
            - Name: {advisor_name}
            - Registration Number: {registration_number or "Not provided"}
            - Database Status: {verification_results["verification_status"]}
            
            PROVIDE DETAILED VERIFICATION ANALYSIS:
            
            1. **CREDENTIAL AUTHENTICITY:**
            - Verify registration number format against SEBI standards
            - Check for common fraudulent patterns in fake registrations
            - Assess completeness of provided information
            
            2. **RED FLAGS ASSESSMENT:**
            - Missing or incomplete registration details
            - Suspicious or unverifiable contact information
            - Claims of unauthorized certifications or awards
            - Social media presence and professional history
            
            3. **REGULATORY COMPLIANCE:**
            - Current SEBI registration status
            - Any disciplinary actions or warnings
            - Compliance with investment advisor regulations
            - Authorized scope of advisory services
            
            4. **RISK EVALUATION:**
            - Overall legitimacy assessment
            - Potential for investor harm
            - Historical complaint patterns
            
            5. **INVESTOR PROTECTION RECOMMENDATIONS:**
            - Specific steps to verify advisor independently
            - Warning signs to watch for in interactions
            - Alternative verification methods
            - Reporting procedures for suspected fraud
            
            6. **SEBI DATABASE CROSS-REFERENCE:**
            - How to verify advisor through official SEBI channels
            - Additional databases to check (BSE, NSE registrations)
            - Professional body memberships (CFP, CFA, etc.)
            
            Format as a comprehensive regulatory compliance report with specific actionable advice.
            """
            
            response = await llm_with_search.acomplete(verification_prompt)
            verification_results["detailed_analysis"] = str(response)
            
        except Exception as e:
            logger.error(f"AI verification analysis error: {e}")
            verification_results["detailed_analysis"] = generate_fallback_verification_analysis(
                advisor_name, registration_number, verification_results
            )
    else:
        verification_results["detailed_analysis"] = generate_fallback_verification_analysis(
            advisor_name, registration_number, verification_results
        )
    
    # Generate specific recommendations based on verification status
    verification_results["recommendations"] = generate_verification_recommendations(
        verification_results["verification_status"],
        verification_results["risk_assessment"]
    )
    
    # Store verification attempt in database
    try:
        db_path = Path("data/regulatory_db/fraud_detection.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO advisor_verifications 
            (advisor_name, registration_number, is_legitimate, verification_status, risk_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            advisor_name,
            registration_number,
            verification_results["is_legitimate"],
            verification_results["verification_status"],
            verification_results["risk_assessment"].get("risk_level", "Unknown")
        ))
        
        conn.commit()
        conn.close()
        
        # Update global state
        fraud_detection_state["advisor_verifications"] += 1
        if not verification_results["is_legitimate"]:
            fraud_detection_state["suspicious_advisors"] += 1
            
    except Exception as e:
        logger.error(f"Verification database error: {e}")
    
    return verification_results

def generate_fallback_verification_analysis(advisor_name: str, registration_number: str, 
                                          verification_results: Dict) -> str:
    """Generate detailed fallback analysis for advisor verification"""
    
    analysis_parts = []
    analysis_parts.append("🔍 SEBI ADVISOR VERIFICATION REPORT")
    analysis_parts.append("=" * 50)
    
    status = verification_results["verification_status"]
    
    analysis_parts.append(f"**ADVISOR:** {advisor_name}")
    if registration_number:
        analysis_parts.append(f"**REGISTRATION NUMBER:** {registration_number}")
    else:
        analysis_parts.append("**REGISTRATION NUMBER:** Not provided (RED FLAG)")
    
    analysis_parts.append(f"**VERIFICATION STATUS:** {status}")
    analysis_parts.append("")
    
    if status == "VERIFIED":
        analysis_parts.append("✅ **VERIFICATION RESULT:** LEGITIMATE ADVISOR")
        analysis_parts.append("The advisor appears in the SEBI database with valid registration.")
        analysis_parts.append("However, always verify current status independently.")
        
    elif status == "SUSPENDED":
        analysis_parts.append("🚨 **CRITICAL WARNING:** SUSPENDED ADVISOR")
        analysis_parts.append("This advisor's registration has been suspended by SEBI.")
        analysis_parts.append("Do NOT engage with this advisor for any investment services.")
        
    elif status == "CANCELLED":
        analysis_parts.append("🚨 **CRITICAL WARNING:** CANCELLED REGISTRATION") 
        analysis_parts.append("This advisor's registration has been cancelled by SEBI.")
        analysis_parts.append("Any investment advice from this person is unauthorized and illegal.")
        
    elif status == "NOT_FOUND":
        analysis_parts.append("🚨 **FRAUD ALERT:** UNREGISTERED ADVISOR")
        analysis_parts.append("This advisor is NOT found in the SEBI database.")
        analysis_parts.append("This is a strong indicator of fraudulent activity.")
        
    else:
        analysis_parts.append("⚠️ **UNABLE TO VERIFY:** Database access issues")
        analysis_parts.append("Manual verification through SEBI website required.")
    
    analysis_parts.append("")
    analysis_parts.append("**INDEPENDENT VERIFICATION STEPS:**")
    analysis_parts.append("1. Visit https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognised=yes")
    analysis_parts.append("2. Search for the advisor name and registration number")
    analysis_parts.append("3. Verify the advisor's current status and authorized services")
    analysis_parts.append("4. Check for any regulatory actions or warnings")
    
    analysis_parts.append("")
    analysis_parts.append("**REGULATORY REQUIREMENTS:**")
    analysis_parts.append("• Only SEBI-registered advisors can provide investment advice for fees")
    analysis_parts.append("• Advisors must provide proper disclosure documents")
    analysis_parts.append("• All advice must include appropriate risk warnings")
    analysis_parts.append("• Advisors cannot guarantee returns or use pressure tactics")
    
    analysis_parts.append("=" * 50)
    
    return "\n".join(analysis_parts)

def generate_verification_recommendations(status: str, risk_assessment: Dict) -> List[str]:
    """Generate specific recommendations based on verification status"""
    
    recommendations = []
    
    if status == "VERIFIED":
        recommendations.extend([
            "✅ Advisor appears legitimate but continue with standard due diligence",
            "📋 Request and review the advisor's disclosure document",
            "💼 Verify the scope of services the advisor is authorized to provide",
            "🔍 Check for any recent regulatory updates or warnings",
            "📱 Independently verify through SEBI website before engaging"
        ])
        
    elif status in ["SUSPENDED", "CANCELLED"]:
        recommendations.extend([
            "🚨 CRITICAL: Do NOT engage with this advisor under any circumstances",
            "📱 Report this advisor to SEBI immediately if they're still operating",
            "⚠️ If you've already invested through them, contact SEBI investor helpline",
            "🛡️ Warn others about this advisor to prevent further fraud",
            "📋 Collect all documentation for potential legal action"
        ])
        
    elif status == "NOT_FOUND":
        recommendations.extend([
            "🚨 FRAUD ALERT: This is likely a fraudulent advisor",
            "📱 Report to SEBI, local police, and cyber crime units immediately",
            "⚠️ Do not share any personal or financial information",
            "🛡️ If money has been transferred, contact your bank immediately",
            "📋 Document all communications for law enforcement"
        ])
        
    else:  # Database error or unknown status
        recommendations.extend([
            "⚠️ Manual verification required due to system limitations",
            "🔍 Visit SEBI website directly to verify advisor credentials",
            "📱 Contact SEBI investor helpline for assistance: 1800-22-7575",
            "💼 Demand proper documentation before proceeding",
            "⚡ Exercise extreme caution until verification is complete"
        ])
    
    return recommendations

async def monitor_social_media_for_fraud(platform: str, search_terms: str) -> Dict:
    """
    Comprehensive social media monitoring for fraudulent investment schemes
    Provides detailed threat assessment and actionable recommendations
    """
    
    monitoring_results = {
        "platform": platform,
        "search_terms": search_terms,
        "threat_level": "Unknown",
        "suspicious_accounts": [],
        "fraud_indicators": [],
        "detailed_analysis": "",
        "recommendations": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Enhanced fraud detection patterns for social media
    social_media_fraud_patterns = {
        "pump_and_dump": [
            "buy before it explodes", "next big thing", "moonshot stock", "rocket to the moon",
            "diamond hands", "ape together strong", "to the moon", "hodl strong"
        ],
        "fake_guru_tactics": [
            "dm for signals", "private group", "exclusive tips", "vip members only",
            "copy my trades", "follow for profits", "millionaire mentor", "trading secrets"
        ],
        "crypto_scams": [
            "send eth get back", "double your crypto", "airdrop opportunity", "free tokens",
            "liquidity mining", "defi yield", "nft investment", "metaverse coins"
        ],
        "forex_fraud": [
            "forex signals", "automated trading", "robot trades", "binary options",
            "copy trading", "prop firm", "funded account", "risk-free trading"
        ],
        "ponzi_schemes": [
            "refer friends", "passive income", "guaranteed returns", "no work required",
            "investment club", "wealth building", "financial freedom", "retire early"
        ],
        "fake_testimonials": [
            "changed my life", "bought new car", "quit my job", "thank you sir",
            "best mentor", "finally rich", "amazing results", "proof of payment"
        ]
    }
    
    detected_fraud_patterns = {}
    total_threat_score = 0
    
    search_terms_lower = search_terms.lower()
    
    # Analyze search terms against fraud patterns
    for category, patterns in social_media_fraud_patterns.items():
        category_matches = []
        for pattern in patterns:
            if pattern in search_terms_lower:
                category_matches.append(pattern)
                total_threat_score += 20
        
        if category_matches:
            detected_fraud_patterns[category] = category_matches
    
    # Simulate detection of suspicious accounts (in real implementation, this would use platform APIs)
    suspicious_accounts = []
    if detected_fraud_patterns:
        # Generate realistic suspicious account names based on detected patterns
        if "pump_and_dump" in detected_fraud_patterns:
            suspicious_accounts.extend(["@moon_trader_pro", "@diamond_hands_king", "@rocket_signals"])
        if "fake_guru_tactics" in detected_fraud_patterns:
            suspicious_accounts.extend(["@trading_guru_vip", "@profit_mentor_365", "@signal_master_dm"])
        if "crypto_scams" in detected_fraud_patterns:
            suspicious_accounts.extend(["@crypto_airdrop_official", "@free_eth_giveaway", "@nft_millionaire"])
    
    # AI-powered comprehensive analysis
    if llm_with_search:
        try:
            social_media_prompt = f"""
            🚨 CONDUCT COMPREHENSIVE SOCIAL MEDIA FRAUD MONITORING 🚨
            
            As a social media fraud detection expert, analyze this monitoring request:
            
            MONITORING PARAMETERS:
            - Platform: {platform}
            - Search Terms: {search_terms}
            - Detected Patterns: {list(detected_fraud_patterns.keys())}
            
            PROVIDE DETAILED THREAT ANALYSIS:
            
            1. **FRAUD SCHEME IDENTIFICATION:**
            - Classify the type of investment fraud being promoted
            - Identify specific tactics used by fraudsters
            - Assess sophistication level of the fraud operation
            
            2. **SOCIAL MEDIA TACTICS ANALYSIS:**
            - Evaluate use of influencer marketing and fake testimonials
            - Analyze group recruitment and community building tactics
            - Assess FOMO (Fear of Missing Out) manipulation techniques
            
            3. **PLATFORM-SPECIFIC RISKS:**
            - {platform} specific fraud vectors and vulnerabilities
            - Common monetization schemes on this platform
            - Platform policy violations and reporting mechanisms
            
            4. **THREAT ACTOR PROFILING:**
            - Identify characteristics of accounts promoting fraudulent schemes
            - Analyze network patterns and coordinated behavior
            - Assess reach and potential victim count
            
            5. **INVESTOR PROTECTION MEASURES:**
            - Immediate actions for platform users to stay safe
            - Red flags to watch for in investment promotions
            - Verification steps before engaging with investment content
            
            6. **REGULATORY IMPLICATIONS:**
            - SEBI social media guidelines violations
            - Advertising standards compliance issues
            - Unauthorized investment advisory activities
            
            Format as a comprehensive social media threat assessment with specific protective actions.
            """
            
            response = await llm_with_search.acomplete(social_media_prompt)
            monitoring_results["detailed_analysis"] = str(response)
            
        except Exception as e:
            logger.error(f"AI social media analysis error: {e}")
            monitoring_results["detailed_analysis"] = generate_fallback_social_media_analysis(
                platform, search_terms, detected_fraud_patterns
            )
    else:
        monitoring_results["detailed_analysis"] = generate_fallback_social_media_analysis(
            platform, search_terms, detected_fraud_patterns
        )
    
    # Determine threat level
    if total_threat_score >= 80:
        monitoring_results["threat_level"] = "Critical"
    elif total_threat_score >= 50:
        monitoring_results["threat_level"] = "High"
    elif total_threat_score >= 20:
        monitoring_results["threat_level"] = "Medium"
    else:
        monitoring_results["threat_level"] = "Low"
    
    # Set results
    monitoring_results["suspicious_accounts"] = suspicious_accounts
    monitoring_results["fraud_indicators"] = list(detected_fraud_patterns.keys())
    monitoring_results["recommendations"] = generate_social_media_recommendations(
        monitoring_results["threat_level"], platform, detected_fraud_patterns
    )
    
    # Store in database
    try:
        db_path = Path("data/regulatory_db/fraud_detection.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO social_media_monitoring 
            (platform, search_terms, findings, threat_level, detailed_analysis)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            platform,
            search_terms,
            json.dumps({
                "suspicious_accounts": suspicious_accounts,
                "fraud_indicators": monitoring_results["fraud_indicators"]
            }),
            monitoring_results["threat_level"],
            monitoring_results["detailed_analysis"]
        ))
        
        conn.commit()
        conn.close()
        
        # Update global state
        fraud_detection_state["social_media_alerts"] += 1
        
    except Exception as e:
        logger.error(f"Social media monitoring database error: {e}")
    
    return monitoring_results

def generate_fallback_social_media_analysis(platform: str, search_terms: str, 
                                          detected_patterns: Dict) -> str:
    """Generate detailed fallback analysis for social media monitoring"""
    
    analysis_parts = []
    analysis_parts.append("🚨 SOCIAL MEDIA FRAUD MONITORING REPORT")
    analysis_parts.append("=" * 50)
    
    analysis_parts.append(f"**PLATFORM:** {platform}")
    analysis_parts.append(f"**SEARCH TERMS:** {search_terms}")
    analysis_parts.append("")
    
    if detected_patterns:
        analysis_parts.append("**FRAUD INDICATORS DETECTED:**")
        
        if "pump_and_dump" in detected_patterns:
            analysis_parts.append("• **Pump & Dump Schemes**: Language associated with coordinated market manipulation")
            
        if "fake_guru_tactics" in detected_patterns:
            analysis_parts.append("• **Fake Trading Gurus**: Promoting private groups and exclusive trading signals")
            
        if "crypto_scams" in detected_patterns:
            analysis_parts.append("• **Cryptocurrency Scams**: Promoting fake airdrops, doubling schemes, or fraudulent tokens")
            
        if "forex_fraud" in detected_patterns:
            analysis_parts.append("• **Forex Fraud**: Automated trading robots and unrealistic profit claims")
            
        if "ponzi_schemes" in detected_patterns:
            analysis_parts.append("• **Ponzi/MLM Schemes**: Referral-based investment programs promising passive income")
            
        if "fake_testimonials" in detected_patterns:
            analysis_parts.append("• **Fabricated Success Stories**: Fake testimonials designed to build credibility")
    else:
        analysis_parts.append("**ASSESSMENT:** Low fraud indicators detected in search terms")
    
    analysis_parts.append("")
    analysis_parts.append("**PLATFORM-SPECIFIC RISKS:**")
    
    if platform.lower() in ["instagram", "facebook"]:
        analysis_parts.append("• Visual content can be easily manipulated to show fake success")
        analysis_parts.append("• Private messaging used to move conversations off-platform")
        analysis_parts.append("• Influencer partnerships may not disclose paid promotions")
        
    elif platform.lower() in ["twitter", "x"]:
        analysis_parts.append("• Viral threads can rapidly spread misinformation")
        analysis_parts.append("• Bot networks amplify fraudulent investment schemes")
        analysis_parts.append("• Character limits encourage oversimplified investment advice")
        
    elif platform.lower() in ["youtube", "tiktok"]:
        analysis_parts.append("• Long-form content allows detailed fraud presentations")
        analysis_parts.append("• Algorithm promotion can amplify harmful content")
        analysis_parts.append("• Comments sections become recruitment grounds")
        
    elif platform.lower() in ["telegram", "whatsapp"]:
        analysis_parts.append("• Private groups enable coordinated fraud operations")
        analysis_parts.append("• End-to-end encryption hinders regulatory oversight")
        analysis_parts.append("• Easy sharing of fraudulent investment materials")
    
    analysis_parts.append("")
    analysis_parts.append("**PROTECTIVE MEASURES:**")
    analysis_parts.append("• Never invest based solely on social media recommendations")
    analysis_parts.append("• Verify all investment advisors through SEBI website")
    analysis_parts.append("• Be suspicious of guaranteed returns or urgent investment opportunities")
    analysis_parts.append("• Report suspicious accounts to platform administrators")
    analysis_parts.append("• Join official investor education groups instead of private trading communities")
    
    analysis_parts.append("=" * 50)
    
    return "\n".join(analysis_parts)

def generate_social_media_recommendations(threat_level: str, platform: str, 
                                        detected_patterns: Dict) -> List[str]:
    """Generate specific recommendations for social media fraud threats"""
    
    recommendations = []
    
    if threat_level in ["Critical", "High"]:
        recommendations.extend([
            f"🚨 CRITICAL: High fraud activity detected on {platform}",
            "📱 Report suspicious accounts to platform administrators immediately",
            "⚠️ Warn your network about these specific fraud tactics",
            "🛡️ Avoid all investment opportunities promoted through this platform",
            "📋 Document evidence for reporting to SEBI and cyber crime units"
        ])
        
        if "pump_and_dump" in detected_patterns:
            recommendations.append("💰 Be especially wary of coordinated stock promotion campaigns")
            
        if "crypto_scams" in detected_patterns:
            recommendations.append("🪙 Never send cryptocurrency to receive 'free' tokens or doubling schemes")
            
    elif threat_level == "Medium":
        recommendations.extend([
            f"⚠️ Moderate fraud risk detected on {platform}",
            "🔍 Exercise extreme caution with investment content on this platform", 
            "📋 Independently verify all investment claims through official sources",
            "💼 Consult SEBI-registered advisors before making investment decisions",
            "📱 Report any suspicious investment promotions encountered"
        ])
        
    elif threat_level == "Low":
        recommendations.extend([
            f"✅ Low fraud indicators for {platform} monitoring",
            "📋 Continue standard due diligence for any investment opportunities",
            "🔍 Stay alert for emerging fraud patterns",
            "💼 Verify advisor credentials before following investment advice"
        ])
        
    # Platform-specific recommendations
    if platform.lower() in ["telegram", "whatsapp"]:
        recommendations.append("🔒 Be especially cautious of private investment groups")
    elif platform.lower() in ["youtube", "tiktok"]:
        recommendations.append("🎥 Remember that video content can be easily manipulated")
    elif platform.lower() in ["instagram", "facebook"]:
        recommendations.append("📸 Fake lifestyle photos are commonly used in investment scams")
    
    return recommendations

async def analyze_corporate_announcement(company_name: str, announcement_text: str) -> Dict:
    """
    Comprehensive corporate announcement analysis for authenticity and legitimacy
    Provides detailed credibility assessment with professional evaluation
    """
    
    announcement_results = {
        "company_name": company_name,
        "credibility_score": 0.0,
        "authenticity_assessment": "Unknown",
        "detailed_analysis": "",
        "fraud_indicators": [],
        "verification_steps": [],
        "recommendations": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Enhanced fraud indicators for corporate announcements
    announcement_fraud_patterns = {
        "unrealistic_claims": [
            "guaranteed profits", "risk-free investment", "500% returns", "no loss guarantee",
            "exclusive opportunity", "limited time only", "revolutionary breakthrough"
        ],
        "credibility_issues": [
            "urgent action required", "act now", "limited seats", "closing soon",
            "invite only", "secret formula", "proprietary technology"
        ],
        "fake_partnerships": [
            "government backing", "ministry approved", "rbi endorsed", "sebi certified",
            "unicorn partnership", "celebrity endorsed", "billionaire backing"
        ],
        "manipulative_language": [
            "life-changing opportunity", "retire early", "passive income", "financial freedom",
            "wealth creation", "millionaire maker", "get rich quick"
        ],
        "missing_disclosures": [
            # Check for absence of proper disclaimers
            "no risk warning", "missing disclaimers", "inadequate disclosure"
        ]
    }
    
    detected_fraud_indicators = {}
    credibility_deductions = 0
    base_credibility = 7.0  # Start with neutral credibility
    
    announcement_lower = announcement_text.lower()
    
    # Analyze announcement text against fraud patterns
    for category, patterns in announcement_fraud_patterns.items():
        category_matches = []
        for pattern in patterns:
            if pattern in announcement_lower:
                category_matches.append(pattern)
                credibility_deductions += 1.5
        
        if category_matches:
            detected_fraud_indicators[category] = category_matches
    
    # Calculate credibility score
    final_credibility = max(0.0, base_credibility - credibility_deductions)
    announcement_results["credibility_score"] = round(final_credibility, 1)
    announcement_results["fraud_indicators"] = list(detected_fraud_indicators.keys())
    
    # Determine authenticity assessment
    if final_credibility >= 7.0:
        announcement_results["authenticity_assessment"] = "Appears Authentic"
    elif final_credibility >= 5.0:
        announcement_results["authenticity_assessment"] = "Requires Verification"
    elif final_credibility >= 3.0:
        announcement_results["authenticity_assessment"] = "Highly Suspicious"
    else:
        announcement_results["authenticity_assessment"] = "Likely Fraudulent"
    
    # AI-powered comprehensive analysis
    if llm_with_search:
        try:
            announcement_prompt = f"""
            🔍 CONDUCT COMPREHENSIVE CORPORATE ANNOUNCEMENT VERIFICATION 🔍
            
            As a corporate communications and fraud detection expert, analyze this announcement:
            
            COMPANY: {company_name}
            ANNOUNCEMENT: {announcement_text}
            
            PROVIDE DETAILED AUTHENTICITY ANALYSIS:
            
            1. **CORPORATE VERIFICATION:**
            - Verify company registration and legal status
            - Check company's official communication channels
            - Cross-reference with stock exchange filings
            - Validate management team and board information
            
            2. **ANNOUNCEMENT AUTHENTICITY:**
            - Language analysis for corporate communication standards
            - Formatting and presentation quality assessment
            - Legal disclaimer and regulatory compliance review
            - Comparison with company's historical announcements
            
            3. **FINANCIAL CLAIMS VERIFICATION:**
            - Realistic assessment of financial projections
            - Comparison with industry benchmarks
            - Evaluation of business model viability
            - Risk factor disclosure adequacy
            
            4. **REGULATORY COMPLIANCE:**
            - SEBI disclosure requirements compliance
            - Stock exchange listing rule adherence
            - Corporate governance standard compliance
            - Insider trading regulation adherence
            
            5. **RED FLAGS ASSESSMENT:**
            - Unrealistic return promises or guarantees
            - Missing mandatory risk warnings
            - Pressure tactics or urgency creation
            - Unverifiable claims or partnerships
            
            6. **MARKET IMPACT ANALYSIS:**
            - Potential effect on stock price and investor sentiment
            - Likelihood of market manipulation intent
            - Assessment of timing and market conditions
            
            7. **VERIFICATION RECOMMENDATIONS:**
            - Specific steps to independently verify claims
            - Official sources for cross-checking information
            - Warning signs for investors to monitor
            
            Format as a comprehensive corporate announcement verification report with actionable investor guidance.
            """
            
            response = await llm_with_search.acomplete(announcement_prompt)
            announcement_results["detailed_analysis"] = str(response)
            
        except Exception as e:
            logger.error(f"AI announcement analysis error: {e}")
            announcement_results["detailed_analysis"] = generate_fallback_announcement_analysis(
                company_name, announcement_text, detected_fraud_indicators, final_credibility
            )
    else:
        announcement_results["detailed_analysis"] = generate_fallback_announcement_analysis(
            company_name, announcement_text, detected_fraud_indicators, final_credibility
        )
    
    # Generate verification steps and recommendations
    announcement_results["verification_steps"] = generate_announcement_verification_steps(company_name)
    announcement_results["recommendations"] = generate_announcement_recommendations(
        announcement_results["authenticity_assessment"], detected_fraud_indicators
    )
    
    # Store in database
    try:
        db_path = Path("data/regulatory_db/fraud_detection.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO corporate_announcements 
            (company_name, announcement_text, credibility_score, authenticity_assessment, detailed_analysis)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            company_name,
            announcement_text[:1000],
            announcement_results["credibility_score"],
            announcement_results["authenticity_assessment"],
            announcement_results["detailed_analysis"]
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Announcement analysis database error: {e}")
    
    return announcement_results

def generate_fallback_announcement_analysis(company_name: str, announcement_text: str,
                                          detected_indicators: Dict, credibility_score: float) -> str:
    """Generate detailed fallback analysis for corporate announcements"""
    
    analysis_parts = []
    analysis_parts.append("🔍 CORPORATE ANNOUNCEMENT VERIFICATION REPORT")
    analysis_parts.append("=" * 50)
    
    analysis_parts.append(f"**COMPANY:** {company_name}")
    analysis_parts.append(f"**CREDIBILITY SCORE:** {credibility_score}/10")
    analysis_parts.append("")
    
    if detected_indicators:
        analysis_parts.append("**FRAUD INDICATORS DETECTED:**")
        
        if "unrealistic_claims" in detected_indicators:
            analysis_parts.append("• **Unrealistic Claims**: Promises that appear too good to be true for legitimate business")
            
        if "credibility_issues" in detected_indicators:
            analysis_parts.append("• **Credibility Issues**: Use of pressure tactics and urgency language")
            
        if "fake_partnerships" in detected_indicators:
            analysis_parts.append("• **Questionable Endorsements**: Claims of backing that should be independently verified")
            
        if "manipulative_language" in detected_indicators:
            analysis_parts.append("• **Manipulative Language**: Emotional appeal designed to bypass rational decision-making")
    else:
        analysis_parts.append("**ASSESSMENT:** Minimal fraud indicators detected in announcement")
    
    analysis_parts.append("")
    analysis_parts.append("**VERIFICATION REQUIREMENTS:**")
    analysis_parts.append("• Check company registration with Ministry of Corporate Affairs")
    analysis_parts.append("• Verify stock exchange listings and recent filings")
    analysis_parts.append("• Cross-reference with official company website and investor relations")
    analysis_parts.append("• Look for SEBI/regulatory disclosures related to this announcement")
    
    analysis_parts.append("")
    analysis_parts.append("**CORPORATE GOVERNANCE INDICATORS:**")
    analysis_parts.append("• Presence of proper legal disclaimers")
    analysis_parts.append("• Compliance with disclosure requirements")
    analysis_parts.append("• Quality of financial data presentation")
    analysis_parts.append("• Management credibility and track record")
    
    analysis_parts.append("")
    analysis_parts.append("**INVESTOR PROTECTION MEASURES:**")
    analysis_parts.append("• Never make investment decisions based solely on announcements")
    analysis_parts.append("• Always verify through multiple independent sources")
    analysis_parts.append("• Consult with SEBI-registered investment advisors")
    analysis_parts.append("• Monitor for consistent follow-up communications")
    
    analysis_parts.append("=" * 50)
    
    return "\n".join(analysis_parts)

def generate_announcement_verification_steps(company_name: str) -> List[str]:
    """Generate specific verification steps for corporate announcements"""
    
    return [
        f"🏢 Verify {company_name} is registered with Ministry of Corporate Affairs",
        "📊 Check stock exchange listings (BSE/NSE) for official filings",
        "🌐 Visit company's official website for investor relations section",
        "📋 Look for SEBI disclosures related to this announcement",
        "📞 Contact company's investor relations department directly",
        "🔍 Search for independent news coverage of the announcement",
        "⚖️ Verify claims with relevant regulatory authorities",
        "📈 Compare with company's historical performance and announcements"
    ]

def generate_announcement_recommendations(authenticity_assessment: str, 
                                        detected_indicators: Dict) -> List[str]:
    """Generate specific recommendations based on announcement analysis"""
    
    recommendations = []
    
    if authenticity_assessment == "Likely Fraudulent":
        recommendations.extend([
            "🚨 CRITICAL FRAUD ALERT: Do not invest based on this announcement",
            "📱 Report this announcement to SEBI and relevant exchanges immediately",
            "⚠️ This appears to be a sophisticated fraud scheme",
            "🛡️ Warn others about this fraudulent announcement",
            "📋 Collect all related materials for law enforcement"
        ])
        
    elif authenticity_assessment == "Highly Suspicious":
        recommendations.extend([
            "🚨 HIGH RISK: Exercise extreme caution with this announcement",
            "🔍 Conduct thorough independent verification before any investment",
            "📱 Consider reporting suspicious elements to authorities",
            "💼 Consult with multiple SEBI-registered advisors",
            "⚠️ Monitor company closely for additional red flags"
        ])
        
    elif authenticity_assessment == "Requires Verification":
        recommendations.extend([
            "⚠️ CAUTION: Additional verification required before proceeding",
            "📋 Complete all recommended verification steps",
            "💼 Seek professional investment advice",
            "🔍 Wait for independent confirmation of claims",
            "📊 Review company's complete financial history"
        ])
        
    else:  # Appears Authentic
        recommendations.extend([
            "✅ Announcement appears legitimate but continue due diligence",
            "📋 Still complete standard verification procedures",
            "💼 Consult with financial advisors for investment decisions",
            "📊 Review all available company information",
            "🔍 Monitor for consistent follow-up disclosures"
        ])
    
    if detected_indicators:
        recommendations.append("⚡ Pay special attention to the fraud indicators identified in the analysis")
    
    return recommendations

# API Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main homepage with Google Careers-inspired design"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "SEBI Fraud Detection System",
        "subtitle": "Protecting retail investors with AI-powered fraud detection",
        "stats": fraud_detection_state
    })

@app.get("/api/stats")
async def get_system_stats():
    """Get real-time system statistics"""
    uptime = datetime.now() - fraud_detection_state["system_uptime"]
    
    return {
        "total_scans": fraud_detection_state["total_scans"],
        "high_risk_detections": fraud_detection_state["high_risk_detections"],
        "advisor_verifications": fraud_detection_state["advisor_verifications"],
        "social_media_alerts": fraud_detection_state["social_media_alerts"],
        "uptime_hours": round(uptime.total_seconds() / 3600, 1),
        "system_status": "Operational",
        "ai_enabled": AI_ENABLED
    }

@app.post("/api/scan-content", response_model=FraudAnalysisResult)
async def scan_content_endpoint(request: ContentScanRequest):
    """Scan content for fraud indicators"""
    try:
        result = await scan_content_for_fraud(request.content, request.content_type)
        
        return FraudAnalysisResult(
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            detected_patterns=result["detected_patterns"],
            recommendations=result["recommendations"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Content scan error: {e}")
        raise HTTPException(status_code=500, detail=f"Content scan failed: {str(e)}")

@app.post("/api/verify-advisor")
async def verify_advisor_endpoint(request: AdvisorVerificationRequest):
    """Verify advisor credentials"""
    try:
        result = await verify_advisor_credentials(request.advisor_name, request.registration_number)
        return result
        
    except Exception as e:
        logger.error(f"Advisor verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Advisor verification failed: {str(e)}")

@app.post("/api/monitor-social-media")
async def monitor_social_media_endpoint(request: SocialMediaMonitorRequest):
    """Monitor social media for fraudulent content with comprehensive analysis"""
    try:
        # Use enhanced monitoring function
        monitoring_results = await monitor_social_media_for_fraud(
            request.platform, 
            request.search_terms
        )
        
        return monitoring_results
        
    except Exception as e:
        logger.error(f"Social media monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Social media monitoring failed: {str(e)}")

@app.post("/api/analyze-announcement")
async def analyze_announcement_endpoint(request: CorporateAnnouncementRequest):
    """Analyze corporate announcement for authenticity with detailed assessment"""
    try:
        # Use enhanced announcement analysis function
        analysis_results = await analyze_corporate_announcement(
            request.company_name,
            request.announcement_text
        )
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Announcement analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Announcement analysis failed: {str(e)}")

@app.get("/api/recent-scans")
async def get_recent_scans():
    """Get recent fraud detection scans"""
    try:
        db_path = Path("data/regulatory_db/fraud_detection.db")
        conn = sqlite3.connect(str(db_path))
        
        # Get recent content scans
        scans_df = pd.read_sql_query('''
            SELECT * FROM content_scans 
            ORDER BY created_at DESC 
            LIMIT 10
        ''', conn)
        
        # Get recent advisor verifications
        verifications_df = pd.read_sql_query('''
            SELECT * FROM advisor_verifications 
            ORDER BY created_at DESC 
            LIMIT 5
        ''', conn)
        
        conn.close()
        
        return {
            "recent_scans": scans_df.to_dict('records') if not scans_df.empty else [],
            "recent_verifications": verifications_df.to_dict('records') if not verifications_df.empty else []
        }
        
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return {"recent_scans": [], "recent_verifications": []}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Fraud detection dashboard"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Fraud Detection Dashboard",
        "stats": fraud_detection_state
    })

@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    """Content scanning page"""
    return templates.TemplateResponse("scan.html", {
        "request": request,
        "title": "Content Scanner"
    })

@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    """Advisor verification page"""
    return templates.TemplateResponse("verify.html", {
        "request": request,
        "title": "Advisor Verification"
    })

@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request):
    """Social media monitoring page"""
    return templates.TemplateResponse("monitor.html", {
        "request": request,
        "title": "Social Media Monitor"
    })

@app.get("/analyzer", response_class=HTMLResponse)
async def analyzer_page(request: Request):
    """Corporate announcement analyzer page"""
    return templates.TemplateResponse("analyzer.html", {
        "request": request,
        "title": "Announcement Analyzer"
    })

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection system on startup"""
    logger.info("🚀 Starting SEBI Fraud Detection System...")
    
    # Initialize database
    init_database()
    
    # Initialize AI system
    ai_initialized = initialize_ai_system()
    
    if ai_initialized:
        logger.info("✅ System startup complete with AI capabilities")
    else:
        logger.info("⚠️ System startup complete in demo mode (limited AI)")
    
    # Create necessary directories
    Path("data/regulatory_db").mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

if __name__ == "__main__":
    # Run the web application
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
