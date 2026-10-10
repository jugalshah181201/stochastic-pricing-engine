# Results summary

GBM setup: S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0. Black-Scholes call = 10.4506

## Variance reduction at N = 100,000 paths

| Method | Std error | Variance reduction vs plain |
|---|---|---|
| Plain | 0.04658 | 1.0x |
| Antithetic | 0.03289 | 2.0x |
| Control variate | 0.01775 | 6.9x |
| Antithetic + control variate | 0.00871 | 28.6x |

## GBM strike sweep (200,000 paths, antithetic + control variate)

| Strike | Black-Scholes | Monte Carlo | Std error | 95% CI covers BS |
|---|---|---|---|---|
| 70 | 33.5401 | 33.5399 | 0.0015 | yes |
| 75 | 28.9744 | 28.9728 | 0.0018 | yes |
| 80 | 24.5888 | 24.5887 | 0.0019 | yes |
| 85 | 20.4693 | 20.4714 | 0.0019 | yes |
| 90 | 16.6994 | 16.6974 | 0.0028 | yes |
| 95 | 13.3465 | 13.3435 | 0.0045 | yes |
| 100 | 10.4506 | 10.4508 | 0.0061 | yes |
| 105 | 8.0214 | 8.0161 | 0.0064 | yes |
| 110 | 6.0401 | 6.0421 | 0.0050 | yes |
| 115 | 4.4666 | 4.4706 | 0.0033 | yes |
| 120 | 3.2475 | 3.2511 | 0.0026 | yes |
| 125 | 2.3243 | 2.3228 | 0.0030 | yes |
| 130 | 1.6396 | 1.6383 | 0.0038 | yes |

13 of 13 intervals cover the Black-Scholes value (about 95% expected).

## Euler-Maruyama bias (2,000,000 antithetic paths)

| Steps | dt | Price | MC minus BS | Std error |
|---|---|---|---|---|
| 1 | 1.0000 | 10.2046 | -0.2460 | 0.0055 |
| 2 | 0.5000 | 10.3614 | -0.0892 | 0.0064 |
| 4 | 0.2500 | 10.4161 | -0.0345 | 0.0068 |
| 8 | 0.1250 | 10.4356 | -0.0150 | 0.0071 |
| 16 | 0.0625 | 10.4405 | -0.0101 | 0.0072 |
| 32 | 0.0312 | 10.4494 | -0.0012 | 0.0073 |
| 64 | 0.0156 | 10.4525 | +0.0019 | 0.0073 |
| 128 | 0.0078 | 10.4455 | -0.0051 | 0.0073 |

Exact sampling (1 step): error -0.0013 (std error 0.0074).

## OU options (x0=90, theta=100, kappa=2, sigma=15, r=3%, T=1; 400,000 paths)

| Strike | OU closed form | OU Monte Carlo | Std error | CI covers | BS misapplied |
|---|---|---|---|---|---|
| 80 | 18.1096 | 18.1098 | 0.0004 | yes | 13.7949 |
| 85 | 13.3372 | 13.3384 | 0.0011 | yes | 10.2473 |
| 90 | 8.8268 | 8.8317 | 0.0024 | no | 7.3138 |
| 95 | 4.9860 | 4.9865 | 0.0042 | yes | 5.0167 |
| 100 | 2.2678 | 2.2635 | 0.0047 | yes | 3.3114 |
| 105 | 0.7860 | 0.7803 | 0.0032 | yes | 2.1078 |
| 110 | 0.1983 | 0.1953 | 0.0016 | yes | 1.2970 |
| 115 | 0.0352 | 0.0346 | 0.0006 | yes | 0.7737 |
| 120 | 0.0043 | 0.0043 | 0.0002 | yes | 0.4487 |

8 of 9 intervals cover the closed-form value.
