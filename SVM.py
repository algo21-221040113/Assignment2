import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import  configparser
import tushare as ts
from jqdatasdk import *
import  configparser

config = configparser.ConfigParser()
config.read('../config.ini')
token = config.get('tushare', 'token')

ts.set_token(token)

origin_daily_300 = ts.pro_bar(ts_code='000300.SH', asset='I').sort_values('trade_date')
origin_daily_300

# origin_weekly_300 = ts.pro_bar(ts_code='000300.SH', freq='W', asset='I').sort_values('trade_date')
# origin_weekly_300


config = configparser.ConfigParser()
config.read('../config.ini')
user, passwd = config.get('joinquant', 'user'), config.get('joinquant', 'passwd')

origin_weekly_300 = get_bars( security = '000300.XSHG',
          count = 215,
          unit = '1w',
          fields=['date','open','high','low','close', 'volume', 'money'],
          include_now=True,
          end_dt='2010-5-21',
          fq_ref_date=None,df=True)



def calc_indicators(mkt_data):
    indi_cols = ['high', 'low', 'pct_chg', 'pct_chg_pre1', 'pct_chg_pre2', 'money_ma4', 'money_pre1']
    # high

    # low

    # pct_chg
    mkt_data['pct_chg'] = 100 * (mkt_data.close / mkt_data.close.shift(1) - 1)
    # pct_chg_pre1
    mkt_data['pct_chg_pre1'] = mkt_data['pct_chg'].shift(1)
    # pct_chg_pre2
    mkt_data['pct_chg_pre2'] = mkt_data['pct_chg'].shift(2)

    if 'money' in mkt_data.columns:
        # money

        # money_ma4
        mkt_data['money_ma4'] = mkt_data['money'].rolling(4).mean()
        # money_pre1
        mkt_data['money_pre1'] = mkt_data['money'].shift(1)
    elif 'amount' in mkt_data.columns:
        # amount

        # amount_ma20
        mkt_data['amount_ma20'] = mkt_data['amount'].rolling(20).mean()
        # money_pre1
        mkt_data['amount_pre1'] = mkt_data['amount'].shift(1)
    return mkt_data

def train_svm_model(X_train, y_train):
    # X为全部列， Y为下一期的pct_chg
    svm_clf = Pipeline(( ("scaler", StandardScaler()),
                         ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
    # 下周上涨为1，否则为0
    svm_clf.fit( X_train, (y_train>0).astype(int) )
    return svm_clf

def rolling_calc_signal(mkt_data,
                        X_cols=['close','open','high','low','pre_close','change','pct_chg','vol','amount'],
                        y_col='pct_chg',
                        N = 20,):
    signals = np.empty(shape=mkt_data[y_col].shape)
    print('Start Rolling Predict...')
    print('N={}, X.shape[0]='.format(N, mkt_data.shape[0]))
    for i in range(len(mkt_data)-N):
        # 每次取N+1行，并用前N行进行训练，最后一行用于预测:
        tmp_data = mkt_data[i:i+N+1].copy()
        # 训练
        X_train = tmp_data[X_cols].values[:-1]
        y_train = tmp_data[y_col].values[1:]
        model = train_svm_model(X_train, y_train)
        # 预测
        X_predict = tmp_data[X_cols].values[[-1]]
        y_predict = model.predict(X_predict)
        signals[i+N] = y_predict[0]
    mkt_data['signal'] = signals
    print('Finish Rolling Predict~')
    return mkt_data

def calc_position(mkt_data, allow_signal_shift=False):
    # 信号是否会出现多空翻转，还是代表恢复空仓
    mkt_data['signal_last'] = mkt_data['signal'].shift(1)
    if allow_signal_shift:
        mkt_data['position'] = mkt_data['signal'].fillna(method='ffill').shift(1).fillna(0)
    else:
        mkt_data['position'] = mkt_data['signal']
    return mkt_data


def statistic_performance(mkt_data):
    position = mkt_data['position']

    # 序列型特征 ----------------------------------
    # 持仓收益
    hold_r = mkt_data['pct_chg'] / 100 * position
    # 持仓胜负
    hold_win = hold_r > 0
    # 持仓净值曲线
    hold_cumu_r = (1 + hold_r).cumprod() - 1
    # 回撤
    drawdown = (hold_cumu_r.cummax() - hold_cumu_r) / (1 + hold_cumu_r).cummax()
    # 超额收益
    ex_hold_r = hold_r - 0.03 / 250

    mkt_data['hold_r'] = hold_r
    mkt_data['hold_win'] = hold_win
    mkt_data['hold_cumu_r'] = hold_cumu_r
    mkt_data['drawdown'] = drawdown
    mkt_data['ex_hold_r'] = ex_hold_r

    # 数值型特征 ----------------------------------
    # 累计收益
    v_hold_cumu_r = hold_cumu_r.tolist()[-1]
    # 多仓次数， 多仓胜率， 多仓平均持有期
    # 空仓次数， 空仓胜率， 空仓平均持有期
    v_pos_hold_times = 0
    v_neg_hold_times = 0
    v_pos_hold_win_times = 0
    v_neg_hold_win_times = 0
    v_pos_hold_period = 0
    v_neg_hold_period = 0
    v_pos_hold_win_period = 0
    v_neg_hold_win_period = 0
    for w, r, pre_pos, pos in zip(hold_win, hold_r, position.shift(1), position):
        # 有换仓（先结算上一次持仓，再初始化本次持仓）
        if pre_pos != pos:
            # 判断pre_pos非空：若为空则是循环的第一次，此时无需结算，直接初始化持仓即可
            if pre_pos == pre_pos:
                # 结算上一次持仓
                if pre_pos > 0:
                    v_pos_hold_times += 1
                    v_pos_hold_period += tmp_hold_period
                    v_pos_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r > 0:
                        v_pos_hold_win_times += 1
                elif pre_pos < 0:
                    v_neg_hold_times += 1
                    v_neg_hold_period += tmp_hold_period
                    v_neg_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r > 0:
                        v_neg_hold_win_times += 1
            # 初始化本次持仓
            tmp_hold_r = r
            tmp_hold_period = 0
            tmp_hold_win_period = 0
        else:  # 未换仓
            if abs(pos) > 0:
                tmp_hold_period += 1
                if r > 0:
                    tmp_hold_win_period += 1
                if abs(r) > 0:
                    tmp_hold_r = (1 + tmp_hold_r) * (1 + r) - 1
                    # 最后一次持仓未结束，不纳入统计

    # 日胜率【持仓天数，持仓收益天数】
    # v_hold_period = v_pos_hold_period + v_neg_hold_period
    # v_hold_win_period = v_pos_hold_win_period + v_neg_hold_win_period
    v_hold_period = (abs(position) > 0).sum()
    v_hold_win_period = (hold_r > 0).sum()

    # 最大回撤
    v_max_dd = drawdown.max()
    # 年化收益
    v_annual_ret = pow(1 + v_hold_cumu_r, 250 / len(mkt_data)) - 1
    # 年化标准差

    # 年化夏普
    v_sharpe = np.sqrt(len(ex_hold_r)) * ex_hold_r.mean() / ex_hold_r.std()

    performance_cols = ['累计收益',
                        '多仓次数', '多仓胜率', '多仓平均持有期',
                        '空仓次数', '空仓胜率', '空仓平均持有期',
                        '日胜率', '最大回撤', '年化收益/最大回撤',
                        '年化收益', '年化标准差', '年化夏普'
                        ]
    performance_values = [v_hold_cumu_r,
                          v_pos_hold_times,
                          0 if v_pos_hold_times == 0 else v_pos_hold_win_times / v_pos_hold_times,
                          0 if v_pos_hold_times == 0 else v_pos_hold_period / v_pos_hold_times,
                          v_neg_hold_times,
                          0 if v_neg_hold_times == 0 else v_neg_hold_win_times / v_neg_hold_times,
                          0 if v_neg_hold_times == 0 else v_neg_hold_period / v_neg_hold_times,
                          v_hold_win_period / v_hold_period, v_max_dd, v_annual_ret / v_max_dd,
                          v_annual_ret, 0, v_sharpe
                          ]
    performance_df = pd.DataFrame(performance_values, index=performance_cols)

    return mkt_data, performance_df

# 前20期构建指标时存在空，去掉
origin_daily_300 = calc_indicators(origin_daily_300)
daily_300 = origin_daily_300[20:].copy()

# 设定X和Y，滚动训练+预测 signal
indi_cols = ['close', 'high', 'low', 'pct_chg', 'pct_chg_pre1', 'pct_chg_pre2', 'amount', 'amount_ma20', 'amount_pre1']
daily_300 = rolling_calc_signal(daily_300, N=100,
                                 X_cols=indi_cols,
                                 y_col='pct_chg',)

# 根据signal生成position
daily_300 = calc_position(daily_300, allow_signal_shift=True)
daily_300

# 评价和展现
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>='20060818' and x<='20100521')])
result_daily_300, performance_daily_300 = statistic_performance(daily_300)

print(performance_daily_300)

# 前3期构建指标时存在空，去掉
origin_weekly_300 = calc_indicators(origin_weekly_300)
weekly_300 = origin_weekly_300[3:].copy()

# 设定X和Y，滚动训练+预测 signal
indi_cols = ['open','close', 'high', 'low',
             'pct_chg', 'pct_chg_pre1', 'pct_chg_pre2',
             'money', 'money_ma4', 'money_pre1']
weekly_300 = rolling_calc_signal( weekly_300, N=20,
                                  X_cols=indi_cols,
                                  y_col='pct_chg',)
# 根据signal生成position
weekly_300 = calc_position(weekly_300, allow_signal_shift=True)

# 评价和展现
weekly_300['trade_date'] = weekly_300['date'].apply(lambda x: str(x).replace('-', ''))
#result_weekly_300, performance_df = statistic_performance(weekly_300[weekly_300['trade_date'].apply(lambda x: x>='20060818' and x<='20100521')])
result_daily_300, performance_weekly_300 = statistic_performance(daily_300)


print(performance_weekly_300)