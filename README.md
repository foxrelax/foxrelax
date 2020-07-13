# foxrelax

## Project description

* it’s easy to use because most of the data returned are pandas DataFrame objects
* we have our own data server, efficient and stable operation
* free stock market data
* friendly to machine learning and data mining

## Target Users

* learners of financial data analysis with pandas/NumPy
* financial market analyst
* financial data analysis enthusiasts
* quanters who are interested in stock market

## Installation

```bash
pip install foxrelax
```

## Upgrade

```bash
pip install foxrelax –upgrade
```

## Quick Start

```python
import foxrelax as relax

resp = relax.stock_daily(symbol='000049.SZ',
                         start_date='2018-01-01',
                         end_date='2019-11-28')


print('code={0} message={1}'.format(resp.code, resp.message))
print(resp.result)
```

return:

```text
code=0 message=success

        symbol  trade_date   open   high    low  close  pre_close  \
0    000049.SZ  2018-01-02  39.91  40.18  39.22  39.78      39.59
1    000049.SZ  2018-01-03  39.90  40.59  39.71  40.56      39.78
2    000049.SZ  2018-01-04  41.61  43.18  41.61  42.65      40.56
3    000049.SZ  2018-01-05  42.00  42.45  41.51  42.07      42.65
4    000049.SZ  2018-01-08  42.08  42.38  41.32  41.49      42.07
5    000049.SZ  2018-01-09  41.50  41.60  40.60  40.80      41.49
...
458  000049.SZ  2019-11-21  36.53  36.77  36.00  36.50      36.81
459  000049.SZ  2019-11-22  36.44  37.80  36.30  36.68      36.50
460  000049.SZ  2019-11-25  36.30  36.48  34.93  35.16      36.68
461  000049.SZ  2019-11-26  35.28  35.89  35.00  35.76      35.16
462  000049.SZ  2019-11-27  35.69  37.27  35.40  36.72      35.76
463  000049.SZ  2019-11-28  36.72  38.48  36.50  37.49      36.72

     price_change  pct_change    volume      money  
0            0.19      0.4800   3197416  126769960  
1            0.78      1.9600   3913276  157416236  
2            2.09      5.1500   8300770  352188889  
3           -0.58     -1.3600   4260338  178822929  
4           -0.58     -1.3800   3569519  148331795  
5           -0.69     -1.6600   3138719  128433609  
...
458         -0.31     -0.8422   4079903  148147523  
459          0.18      0.4932   8250452  306343236  
460         -1.52     -4.1439   6989292  247534854  
461          0.60      1.7065   4304988  153468524  
462          0.96      2.6846   7701535  281834519  
463          0.77      2.0969  10463935  394639040  
```
