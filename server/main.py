from flask import Flask, Blueprint, render_template, session, jsonify, request, redirect, url_for, flash, g
from flask_login import login_required, current_user, login_user, logout_user
#from flask_environments import Environments
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import pytz
import re
from tickers_sorted import *
from tickers_sorted_tw import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from cvxopt import matrix, solvers
from cvxopt.blas import dot
from cvxopt.solvers import qp

main = Flask(__name__)
#main = Blueprint('main', __name__)

main.config['SECRET_KEY'] = os.urandom(30)

#env = Environments(main)

# Source: https://uniwebsidad.com/libros/explore-flask/chapter-8/custom-filters
@main.template_filter('my_substitution')
def my_substitution(string):
    return re.sub(r'@[a-zA-Z0-9_\-\.]+', r'', string)

#env.filters['my_substitution'] = my_substitution
                                    



def connect_db():
    sql = sqlite3.connect('strategy.db')
    sql.row_factory = sqlite3.Row
    return sql 

def get_db():
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

@main.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/create_strategy', methods=['GET', 'POST'])
def create_strategy():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    if request.method == 'POST':
        strategy_name = request.form['strategy_name']
        if strategy_name == '':
            flash('請取一個名字', 'danger')
            return render_template('create_strategy.html', asset_candidates=asset_candidates)
        tickers = sorted(list(set(request.form.getlist('asset_ticker'))))
        print('The list of assets: ', tickers)

        
        # Turn off progress printing
        solvers.options['show_progress'] = False
        
        start_dates = [datetime(2015, 1, 1),
                       datetime(2015, 4, 1),
                       datetime(2015, 7, 1),
                       datetime(2015, 10, 1),
                       datetime(2016, 1, 1),
                       datetime(2016, 4, 1),
                       datetime(2016, 7, 1),
                       datetime(2016, 10, 1),
                       datetime(2017, 1, 1),
                       datetime(2017, 4, 1),
                       datetime(2017, 7, 1),
                       datetime(2017, 10, 1),
                       datetime(2018, 1, 1),
                       datetime(2018, 4, 1),
                       datetime(2018, 7, 1),
                       datetime(2018, 10, 1),
                       datetime(2019, 1, 1),
                       datetime(2019, 4, 1),
                       datetime(2019, 7, 1),
                       datetime(2019, 10, 1),
                       datetime(2020, 1, 1),
                       datetime(2020, 4, 1),
                       datetime(2020, 7, 1),
                       datetime(2020, 10, 1) ]
        
        all_data = pd.read_csv('data_for_trading_platform.csv')
        all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
        def stockpri(ticker, start, end):
            data = all_data[ (all_data['Ticker']==ticker) & (all_data['Date']>=start) & (all_data['Date']<=end) ]
            data.set_index('Date', inplace=True)
            data = data['Adj Close']
            return data
        
        def result(weight):
            sigma = np.sqrt(np.dot(weight, np.dot(log_returns.cov()*252, weight.T)))
            profit = np.dot(weight, np.exp(log_returns.mean()*252) - 1) + 1
            return np.array([sigma, profit, profit/sigma])
        
        def sigma(weight):
            return result(weight)[0]
        
        portfolio_value = pd.Series([100])
        optimal_weights = None
        
        for i in range(len(start_dates)-3):
        
            ### Take 6 months to backtest ###
        
            start = start_dates[i]
            end   = start_dates[i+2]
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            if log_returns.empty:
                continue
            
            # Markowitz frontier
            profit = np.linspace(0., 3., 100)
            frontier = []
            w = []
            for p in profit:
                # Problem data.
                n = len(tickers)
                S = matrix(log_returns.cov().values*252)
                pbar = matrix(0.0, (n,1))
                # Gx <= h
                G = matrix(0.0, (2*n,n))
                G[::(2*n+1)] = 1.0
                G[n::(2*n+1)] = -1.0
                # h = matrix(1.0, (2*n,1))
                h = matrix(np.concatenate((0.5*np.ones((n,1)), -0.03*np.ones((n,1))), axis=0))
                A = matrix(np.concatenate((np.ones((1,n)), np.exp(log_returns.mean()*252).values.reshape((1,n))), axis=0))
                b = matrix([1, p], (2, 1))
                
                # Compute trade-off.
                res = qp(S, -pbar, G, h, A, b)
            
                if res['status'] == 'optimal':
                    res_weight = res['x']
                    s = math.sqrt(dot(res_weight, S*res_weight))
                    frontier.append(np.array([p, s]))
                    w.append(res_weight)
            frontier = np.array(frontier)
            if frontier.shape == (0,):
                continue
            x = np.array(frontier[:, 0])
            y = np.array(frontier[:, 1])
        
            frontier_sharpe_ratios = np.divide(x-1, y)
            optimal_portfolio_index = np.argmax(frontier_sharpe_ratios)
            optimal_weights = w[optimal_portfolio_index]
            
        
            ### paper trade on the next three months ###
        
            start = start_dates[i+2]
            end   = start_dates[i+3]
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()
        
            portfolio_value_new_window = portfolio_value.iloc[-1].item() * pd.Series(np.dot(returns, optimal_weights).cumprod())
            portfolio_value_new_window.index = returns.index
            portfolio_value = portfolio_value.append(portfolio_value_new_window) 
            
        if optimal_weights == None:
            sharpe_ratio = avg_annual_return = annual_volatility = max_drawdown = 0
            optimal_weights = [0, ]*len(tickers)
            db = get_db()
            cur = db.cursor()
            create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M') 
            cur.execute('insert into strategy (strategy_name, author, create_date, sharpe_ratio, return, volatility, max_drawdown, tw) values (?,?,?,?,?,?,?,?)', 
                       [strategy_name, 
                        session['USERNAME'], 
                        create_date,
                        sharpe_ratio,
                        avg_annual_return,
                        annual_volatility,
                        max_drawdown,
                        0
                       ] )
            db.commit()

            # record the list of tickers into database
            strategy_id = db.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
            print('Strategy_id ' + str(strategy_id) + ' optimization fails.')
            for i in range(len(tickers)):
                cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (?, ?, ?)', [strategy_id, tickers[i], optimal_weights[i]])
                db.commit()

            db.close()    
            flash('無資料或無法畫出馬可維茲邊界，請換一個組合', 'danger')
            return render_template('create_strategy.html', asset_candidates=asset_candidates)
        
        avg_annual_return = np.exp(np.log(portfolio_value.pct_change() + 1).mean() * 252) - 1
        annual_volatility = portfolio_value.pct_change().std() * math.sqrt(252)
        sharpe_ratio = avg_annual_return/annual_volatility
        max_drawdown = - np.amin(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value)) - 1)
        
        print('The last optimal weights are\n', optimal_weights)
        print('Sharpe ratio: ', sharpe_ratio, ', Return: ', avg_annual_return, ', Volatility: ', annual_volatility, ', Maximum Drawdown: ', max_drawdown)
        
        plt.plot(portfolio_value.iloc[1:])

        db = get_db()
        cur = db.cursor()
        create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M') 
        cur.execute('insert into strategy (strategy_name, author, create_date, sharpe_ratio, return, volatility, max_drawdown, tw) values (?,?,?,?,?,?,?,?)', 
                   [strategy_name, 
                    session['USERNAME'], 
                    create_date,
                    sharpe_ratio,
                    avg_annual_return,
                    annual_volatility,
                    max_drawdown,
                    0
                   ] )
        db.commit()
        flash('回測已完成，請到討論區查看結果。', 'success')

        # record the list of tickers into database
        strategy_id = db.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
        plt.savefig('static/img/portfolio_values/'+str(strategy_id)+'.png')
        plt.close()
        print('Strategy_id ' + str(strategy_id) + ' optimization succeeds.')
        for i in range(len(tickers)):
            cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (?, ?, ?)', [strategy_id, tickers[i], optimal_weights[i]])
            db.commit()

        db.close()    
    return render_template('create_strategy.html', asset_candidates=asset_candidates)


@main.route('/create_strategy_tw', methods=['GET', 'POST'])
def create_strategy_tw():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    if request.method == 'POST':
        strategy_name = request.form['strategy_name']
        if strategy_name == '':
            flash('請取一個名字', 'danger')
            return render_template('create_strategy_tw.html', asset_candidates_tw=asset_candidates_tw)
        tickers = sorted(list(set(request.form.getlist('asset_ticker'))))
        print('The list of assets: ', tickers)

        
        # Turn off progress printing
        solvers.options['show_progress'] = False
        
        start_dates = [datetime(2015, 1, 1),
                       datetime(2015, 4, 1),
                       datetime(2015, 7, 1),
                       datetime(2015, 10, 1),
                       datetime(2016, 1, 1),
                       datetime(2016, 4, 1),
                       datetime(2016, 7, 1),
                       datetime(2016, 10, 1),
                       datetime(2017, 1, 1),
                       datetime(2017, 4, 1),
                       datetime(2017, 7, 1),
                       datetime(2017, 10, 1),
                       datetime(2018, 1, 1),
                       datetime(2018, 4, 1),
                       datetime(2018, 7, 1),
                       datetime(2018, 10, 1),
                       datetime(2019, 1, 1),
                       datetime(2019, 4, 1),
                       datetime(2019, 7, 1),
                       datetime(2019, 10, 1),
                       datetime(2020, 1, 1),
                       datetime(2020, 4, 1),
                       datetime(2020, 7, 1),
                       datetime(2020, 10, 1) ]
        
        all_data = pd.read_csv('data_for_trading_platform_tw.csv')
        all_data['Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
        
        def stockpri(ticker, start, end):
            data = all_data[ (all_data['Ticker']==ticker) & (all_data['Date']>=start) & (all_data['Date']<=end) ]
            data.set_index('Date', inplace=True)
            data = data['Adj Close']
            return data
        
        def result(weight):
            sigma = np.sqrt(np.dot(weight, np.dot(log_returns.cov()*252, weight.T)))
            profit = np.dot(weight, np.exp(log_returns.mean()*252) - 1) + 1
            return np.array([sigma, profit, profit/sigma])
        
        def sigma(weight):
            return result(weight)[0]
        
        portfolio_value = pd.Series([100])
        optimal_weights = None
        
        for i in range(len(start_dates)-3):
        
            ### Take 6 months to backtest ###
        
            start = start_dates[i]
            end   = start_dates[i+2]
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()

            if log_returns.empty:
                continue
            
            # Markowitz frontier
            profit = np.linspace(0., 3., 100)
            frontier = []
            w = []
            for p in profit:
                # Problem data.
                n = len(tickers)
                S = matrix(log_returns.cov().values*252)
                pbar = matrix(0.0, (n,1))
                # Gx <= h
                G = matrix(0.0, (2*n,n))
                G[::(2*n+1)] = 1.0
                G[n::(2*n+1)] = -1.0
                # h = matrix(1.0, (2*n,1))
                h = matrix(np.concatenate((0.5*np.ones((n,1)), -0.03*np.ones((n,1))), axis=0))
                A = matrix(np.concatenate((np.ones((1,n)), np.exp(log_returns.mean()*252).values.reshape((1,n))), axis=0))
                b = matrix([1, p], (2, 1))
                
                # Compute trade-off.
                res = qp(S, -pbar, G, h, A, b)
            
                if res['status'] == 'optimal':
                    res_weight = res['x']
                    s = math.sqrt(dot(res_weight, S*res_weight))
                    frontier.append(np.array([p, s]))
                    w.append(res_weight)
            frontier = np.array(frontier)
            if frontier.shape == (0,):
                continue
            x = np.array(frontier[:, 0])
            y = np.array(frontier[:, 1])
        
            frontier_sharpe_ratios = np.divide(x-1, y)
            optimal_portfolio_index = np.argmax(frontier_sharpe_ratios)
            optimal_weights = w[optimal_portfolio_index]
            
        
            ### paper trade on the next three months ###
        
            start = start_dates[i+2]
            end   = start_dates[i+3]
        
            data = pd.DataFrame({ ticker: stockpri(ticker, start, end) for ticker in tickers })
            data = data.dropna()
            
            returns = data.pct_change() + 1
            returns = returns.dropna()
            log_returns = np.log(data.pct_change() + 1)
            log_returns = log_returns.dropna()
        
            portfolio_value_new_window = portfolio_value.iloc[-1].item() * pd.Series(np.dot(returns, optimal_weights).cumprod())
            portfolio_value_new_window.index = returns.index
            portfolio_value = portfolio_value.append(portfolio_value_new_window) 
            
        if optimal_weights==None:
            sharpe_ratio = avg_annual_return = annual_volatility = max_drawdown = 0
            optimal_weights = [0, ]*len(tickers)
            db = get_db()
            cur = db.cursor()
            create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M') 
            cur.execute('insert into strategy (strategy_name, author, create_date, sharpe_ratio, return, volatility, max_drawdown, tw) values (?,?,?,?,?,?,?,?)', 
                       [strategy_name, 
                        session['USERNAME'], 
                        create_date,
                        sharpe_ratio,
                        avg_annual_return,
                        annual_volatility,
                        max_drawdown,
                        1
                       ] )
            db.commit()

            # record the list of tickers into database
            strategy_id = db.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
            print('Strategy_id ' + str(strategy_id) + ' optimization fails.')
            for i in range(len(tickers)):
                cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (?, ?, ?)', [strategy_id, tickers[i], optimal_weights[i]])
                db.commit()

            db.close()    
            flash('無資料或無法畫出馬可維茲邊界，請換一個組合', 'danger')
            return render_template('create_strategy_tw.html', asset_candidates_tw=asset_candidates_tw)
        
        avg_annual_return = np.exp(np.log(portfolio_value.pct_change() + 1).mean() * 252) - 1
        annual_volatility = portfolio_value.pct_change().std() * math.sqrt(252)
        sharpe_ratio = avg_annual_return/annual_volatility
        max_drawdown = - np.amin(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value)) - 1)
        
        print('The last optimal weights are\n', optimal_weights)
        print('Sharpe ratio: ', sharpe_ratio, ', Return: ', avg_annual_return, ', Volatility: ', annual_volatility, ', Maximum Drawdown: ', max_drawdown)
        
        plt.plot(portfolio_value.iloc[1:])

        db = get_db()
        cur = db.cursor()
        create_date = datetime.strftime(datetime.now() + timedelta(hours=8), '%Y/%m/%d %H:%M') 
        cur.execute('insert into strategy (strategy_name, author, create_date, sharpe_ratio, return, volatility, max_drawdown, tw) values (?,?,?,?,?,?,?,?)', 
                   [strategy_name, 
                    session['USERNAME'], 
                    create_date,
                    sharpe_ratio,
                    avg_annual_return,
                    annual_volatility,
                    max_drawdown,
                    1
                   ] )
        db.commit()
        flash('回測已完成，請到討論區查看結果。', 'success')

        # record the list of tickers into database
        strategy_id = db.execute('select * from strategy where create_date=?', [create_date]).fetchone()['strategy_id']
        plt.savefig('static/img/portfolio_values/'+str(strategy_id)+'.png')
        plt.close()
        print('Strategy_id ' + str(strategy_id) + ' optimization succeeds.')
        for i in range(len(tickers)):
            cur.execute('insert into assets_in_strategy (strategy_id, asset_ticker, weight) values (?, ?, ?)', [strategy_id, tickers[i], optimal_weights[i]])
            db.commit()

        db.close()    
    return render_template('create_strategy_tw.html', asset_candidates_tw=asset_candidates_tw)


@main.route('/login')
def login():
    return render_template('login.html')


@main.route('/login', methods=['POST'])
def login_post():
    password = request.form.get('password')
    username = request.form.get('username')

    db = get_db()
    sql_result = db.execute('select * from user where username=?', [username]).fetchone()

    if (not sql_result) or (not check_password_hash(sql_result['password'], password)):
        flash('使用者代號不對或密碼不對，請再試一次。', 'danger')
        return redirect('/login')

    print(sql_result['username'], sql_result['user_id'])
    session['login'] = True
    session['user_id'] = sql_result['user_id']
    session['USERNAME'] = sql_result['username']
    db.close()    
    return redirect('/')


@main.route('/logout')
#@login_required
def logout():
    #logout_user()
    session['user_id'] = -1
    session['USERNAME'] = None
    session['login'] = False
    return redirect('/login')


@main.route('/signup')
def signup():
    return render_template('signup.html')


@main.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    db = get_db()
    sql_result = db.execute('select * from user where username=?', [username]).fetchone()
    if sql_result:  # if a user is found, we want to redirect back to signup page so user can try again
        flash('這個Email地址已經被使用', 'danger')
        db.close()    
        return redirect('/signup')

    if not (password == confirm_password):
        flash('所輸入兩次密碼不同', 'danger')
        db.close()    
        return redirect('/signup')

    # create new user with the form data. Hash the password so plaintext version isn't saved.
    db.execute('insert into user (username, password) values (?, ?)', [username, generate_password_hash(password)])
    db.commit()

    print('registered')
    flash('已成功註冊', 'success')

    db.close()    
    return redirect('/login')


@main.route('/forum')
def forum_index():
    db = get_db()
    data = db.execute('select * from strategy order by strategy_id desc').fetchall()

    content_list = []
    for d in data:
        content_list.append({
            "id": d['strategy_id'],
            "time": d['create_date'],
            "user_id": None,
            "user_email": None,
            "user_name": d['author'],
            "comment": None,
            "title": d['strategy_name'],
            "video_id": None
        })

    return_data = {
        "count": len(data),
        "content": content_list
    }

    db.close()    
    return render_template('forum.html', forum_data=return_data)


@main.route('/post_page', methods=['GET'])
# @login_required
def post_page():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    post_id = int(request.values.get('post_id'))

    db = get_db()
    strategy_content_list = db.execute('select * from strategy where strategy_id=?', [post_id]).fetchone()
    asset_list = db.execute('select * from assets_in_strategy where strategy_id=?', [post_id]).fetchall()
    comment_list = db.execute('select * from comment where strategy_id=?', [post_id]).fetchall()

    print(asset_list)

    return_data = {
        "strategy_content": strategy_content_list,
        "asset_content": asset_list,
        "comment_content": comment_list,
        "comment_count": len(comment_list)
    }

    db.close()    
    return render_template('post_page.html', data=return_data, strategy_id=str(post_id))


@main.route('/comment', methods=['POST'])
#@login_required
def post_comment_data():
    comment = request.form['comment']
    strategy_id = request.form['strategy_id']
    author = session['USERNAME']
    comment_date = datetime.strftime(datetime.utcnow().replace(tzinfo=pytz.timezone('Asia/Taipei')), '%Y/%m/%d %H:%M') 

    db = get_db()
    db.execute('insert into comment (author, strategy_id, comment, date) values (?, ?, ?, ?)', [author, strategy_id, comment, comment_date])
    db.commit()

    db.close()    
    return redirect('/post_page?post_id='+str(strategy_id))


#@main.route('/profile')
#@login_required
#def profile():
#    return render_template('profile.html', name=current_user.name)


@main.route('/analysis_result')
#@login_required
def analysis_result():
    if not (session.get('USERNAME') and session['USERNAME']):
        flash('使用此功能必須先登入。', 'danger')
        return redirect('/login')
    db = get_db()

    sortby = request.values.get('sortby')
    tw = request.values.get('tw')
    tw_digit = 1 if tw=='true' else 0 if tw=='false' else None

    if sortby == 'default':
        sql_results = db.execute('select * from strategy where tw=? order by strategy_id desc', [tw_digit]).fetchall()
    elif sortby == 'myself':
        sql_results = db.execute('select * from strategy where tw=? and author=? order by strategy_id desc', [tw_digit, session['USERNAME']]).fetchall()
    elif sortby == 'return':
        sql_results = db.execute('select * from strategy where tw=? order by return desc', [tw_digit]).fetchall()
    elif sortby == 'vol':
        sql_results = db.execute('select * from strategy where tw=? order by volatility asc', [tw_digit]).fetchall()
    elif sortby == 'sharpe':
        sql_results = db.execute('select * from strategy where tw=? order by sharpe_ratio desc', [tw_digit]).fetchall()
    elif sortby == 'mdd':
        sql_results = db.execute('select * from strategy where tw=? order by max_drawdown asc', [tw_digit]).fetchall()
    db.close()    
    return render_template('result.html', results=sql_results, tw=tw)





if __name__ == "__main__":
    main.run(host='0.0.0.0', port=80)
