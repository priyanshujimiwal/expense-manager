from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime, timedelta
import os
import json
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expense_tracker.db'
app.config['SECRET_KEY'] = 'your_secret_key_here_change_this'

# Initialize database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ===== DATABASE MODELS =====

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    expenses = db.relationship('Expense', backref='user', lazy=True, cascade='all, delete-orphan')

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200))
    date = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== HELPER FUNCTIONS =====

def get_category_totals(user_id, days=30):
    """Get total spending by category for last N days"""
    start_date = datetime.utcnow() - timedelta(days=days)
    expenses = Expense.query.filter(
        Expense.user_id == user_id,
        Expense.date >= start_date
    ).all()
    
    category_totals = defaultdict(float)
    for expense in expenses:
        category_totals[expense.category] += expense.amount
    
    return dict(category_totals)

def predict_next_month_spending(user_id):
    """Use ML to predict next month's spending by category"""
    expenses = Expense.query.filter_by(user_id=user_id).all()
    
    if len(expenses) < 5:  # Need at least 5 data points
        return None
    
    # Prepare data
    df = pd.DataFrame([
        {
            'date': exp.date,
            'category': exp.category,
            'amount': exp.amount
        }
        for exp in expenses
    ])
    
    df['days_ago'] = (datetime.utcnow() - df['date']).dt.days
    
    predictions = {}
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category].sort_values('days_ago')
        
        if len(cat_data) < 3:
            continue
        
        X = cat_data['days_ago'].values.reshape(-1, 1)
        y = cat_data['amount'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for next month (30 days ahead)
            next_month_pred = model.predict([[0]])[0]
            predictions[category] = max(0, next_month_pred)  # Don't predict negative
        except:
            predictions[category] = cat_data['amount'].mean()
    
    return predictions

def get_savings_opportunities(category_predictions):
    """Calculate where user can save money"""
    if not category_predictions:
        return {}
    
    savings = {}
    avg_spending = sum(category_predictions.values()) / len(category_predictions)
    
    for category, amount in category_predictions.items():
        if amount > avg_spending * 1.2:  # 20% above average
            savings[category] = amount * 0.2  # Suggest 20% reduction
    
    return savings

# ===== ROUTES =====

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            return "Username already exists!", 400
        
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials!", 400
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    expenses = Expense.query.filter_by(user_id=current_user.id).order_by(Expense.date.desc()).all()
    category_totals = get_category_totals(current_user.id, days=30)
    total_spent = sum(category_totals.values())
    
    return render_template('dashboard.html', 
                         expenses=expenses,
                         category_totals=category_totals,
                         total_spent=total_spent)

@app.route('/add-expense', methods=['POST'])
@login_required
def add_expense():
    category = request.form['category']
    amount = request.form['amount']
    description = request.form['description']
    
    expense = Expense(user_id=current_user.id, category=category, amount=float(amount), description=description)
    db.session.add(expense)
    db.session.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/api/pie-chart')
@login_required
def pie_chart_data():
    """API endpoint for pie chart data"""
    category_totals = get_category_totals(current_user.id, days=30)
    
    return jsonify({
        'labels': list(category_totals.keys()),
        'values': list(category_totals.values())
    })

@app.route('/predictions')
@login_required
def predictions():
    """Predictions and savings opportunities page"""
    predictions = predict_next_month_spending(current_user.id)
    
    if not predictions:
        return render_template('predictions.html', 
                             error="Need at least 5 expenses to generate predictions")
    
    savings = get_savings_opportunities(predictions)
    total_predicted = sum(predictions.values())
    total_savings = sum(savings.values())
    
    return render_template('predictions.html',
                         predictions=predictions,
                         savings=savings,
                         total_predicted=total_predicted,
                         total_savings=total_savings)

@app.route('/invest')
@login_required
def invest():
    """Investment recommendations page"""
    predictions = predict_next_month_spending(current_user.id)
    savings = get_savings_opportunities(predictions)
    total_savings = sum(savings.values())
    
    # Sample investment options
    investments = [
        {
            'name': 'Nifty 50 Index Fund',
            'type': 'Mutual Fund',
            'min_investment': 500,
            'expected_return': '12-15%',
            'risk': 'Medium',
            'description': 'Diversified index fund tracking top 50 companies'
        },
        {
            'name': 'ICICI Prudential Growth Fund',
            'type': 'Mutual Fund',
            'min_investment': 1000,
            'expected_return': '14-18%',
            'risk': 'High',
            'description': 'Growth-oriented mutual fund'
        },
        {
            'name': 'HDFC Fixed Deposit',
            'type': 'Fixed Deposit',
            'min_investment': 1000,
            'expected_return': '6-7%',
            'risk': 'Low',
            'description': 'Safe fixed deposit with guaranteed returns'
        },
        {
            'name': 'SBI Smart Savings Fund',
            'type': 'Savings Plan',
            'min_investment': 500,
            'expected_return': '8-10%',
            'risk': 'Low-Medium',
            'description': 'Flexible savings with insurance protection'
        },
        {
            'name': 'Axis Long Term Equity Fund',
            'type': 'Mutual Fund',
            'min_investment': 500,
            'expected_return': '13-17%',
            'risk': 'High',
            'description': 'Long-term growth equity fund'
        }
    ]
    
    return render_template('invest.html',
                         investments=investments,
                         total_savings=total_savings,
                         predicted_savings=total_savings)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


# hello world