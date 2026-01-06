
## noise: Add noise to input data for regularization
- value 0.1f0 seems to work well

- Info: Computing statistics for timespan: test (2024-01-01T00:00:00 to 2024-11-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.185446  0.299396  0.236          0.279843       0.585715   7016
   2 │ Europlatform  0.180555  0.258433  0.213452       0.307132       0.552644   7016
   3 │ F3            0.212536  0.382281  0.29915        0.241168       0.448671   7016

- Info: Computing statistics for timespan: 202401 (2024-01-01T00:00:00 to 2024-02-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.229572  0.40654   0.309121       0.133721       0.223625    745
   2 │ Europlatform  0.204587  0.29528   0.238222       0.128463       0.180394    745
   3 │ F3            0.292365  0.557593  0.425992       0.167699       0.328252    745

- Info: Computing statistics for timespan: training (2021-01-01T00:00:00 to 2023-12-31T23:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias       rmse      mae       relative_bias  scatter_index  count
     │ String        Float32    Float32   Float32   Float32        Float32        Int64
─────┼──────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.0175837  0.290066  0.219079      0.091058        0.335804  26260
   2 │ Europlatform  0.0275382  0.23721   0.18366       0.103152        0.357434  26260
   3 │ F3            0.022574   0.37107   0.278159      0.0789923       0.310153  26257

## stress: Change wind vector u,v to stress components t_x, t_y
- noise value 0.001f0 seems to work well

- Info: Computing statistics for timespan: test (2024-01-01T00:00:00 to 2024-11-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.166822  0.288963  0.223002       0.256973       0.582998   7016
   2 │ Europlatform  0.170925  0.247775  0.201416       0.301276       0.586512   7016
   3 │ F3            0.17616   0.366824  0.280313       0.20815        0.436535   7016

- Info: Computing statistics for timespan: 202401 (2024-01-01T00:00:00 to 2024-02-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.179585  0.347883  0.267449       0.103089       0.19537     745
   2 │ Europlatform  0.1724    0.245643  0.200513       0.113139       0.162021    745
   3 │ F3            0.192403  0.518305  0.376535       0.107141       0.279697    745

- Info: Computing statistics for timespan: training (2021-01-01T00:00:00 to 2023-12-31T23:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias        rmse      mae       relative_bias  scatter_index  count
     │ String        Float32     Float32   Float32   Float32        Float32        Int64
─────┼───────────────────────────────────────────────────────────────────────────────────
   1 │ K13a          0.00649506  0.277464  0.203767       0.075446       0.328758  26260
   2 │ Europlatform  0.0141656   0.223312  0.17221        0.091823       0.372874  26260
   3 │ F3            5.47195e-5  0.345366  0.250066       0.058621       0.300141  26257

## 7to3: Use 7 input features instead of only 3 
- input locations: "K13a" "F3" "Europlatform" "Gannet platform 1" "A121" "D151" "nsb3"
- const regularization_weight=1.0f-4 #1.0e-4
- const input_noise_std=0.10f0

- Info: Computing statistics for timespan: test (2024-01-01T00:00:00 to 2024-11-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ Europlatform  0.165136  0.254712  0.20493        0.330011       0.693784   7016
   2 │ K13a          0.157023  0.277656  0.219738       0.263489       0.637967   7016
   3 │ F3            0.161313  0.345845  0.26618        0.197308       0.450972   7016

- Info: Computing statistics for timespan: 202401 (2024-01-01T00:00:00 to 2024-02-01T00:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias      rmse      mae       relative_bias  scatter_index  count
     │ String        Float32   Float32   Float32   Float32        Float32        Int64
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ Europlatform  0.154282  0.274323  0.215676      0.0870682       0.162138    745
   2 │ K13a          0.175751  0.35937   0.286346      0.0953848       0.203558    745
   3 │ F3            0.205354  0.519043  0.398769      0.111655        0.26885     745

- Info: Computing statistics for timespan: training (2021-01-01T00:00:00 to 2023-12-31T23:00:00)
stats = 3×7 DataFrame
 Row │ station_name  bias         rmse      mae       relative_bias  scatter_index  count
     │ String        Float32      Float32   Float32   Float32        Float32        Int64
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │ Europlatform   0.0174308   0.2344    0.180301       0.115257       0.449558  26260
   2 │ K13a           7.05038e-5  0.27111   0.200511       0.077243       0.359664  26260
   3 │ F3            -0.0265773   0.336099  0.243168       0.03802        0.291812  26257

## 10to11: Use more locations (10) and predict at more locations (11)
- 

- Info: Computing statistics for timespan: test (2024-01-01T00:00:00 to 2024-11-01T00:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias      rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32   Float32   Float32   Float32        Float32        Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     0.142433  0.319895  0.243424       0.198748       0.492217   7016
   2 │ Schiermonnikoog Noord    0.1853    0.290823  0.231646       0.354638       0.71508    7016
   3 │ L91                      0.133709  0.289828  0.223889       0.208842       0.501753   7016
   4 │ wadden eierlandse gat    0.137292  0.275645  0.212717       0.235477       0.545578   7016
   5 │ OS11                     0.225544  0.286521  0.240575       0.592424       1.0892     7016
   6 │ Europlatform             0.161949  0.254429  0.20412        0.321414       0.68106    7016
   7 │ ijmuiden munitiestort 1  0.177615  0.28465   0.223487       0.323562       0.688586   7016
   8 │ K13a                     0.154887  0.288762  0.222741       0.251898       0.6211     7016
   9 │ A121                     0.158791  0.382777  0.292876       0.176257       0.430209   7016
  10 │ eurogeul E13             0.181935  0.264682  0.214368       0.378018       0.781578   7016
  11 │ F3                       0.160904  0.35798   0.276397       0.188342       0.43951    7016

- Info: Computing statistics for timespan: 202401 (2024-01-01T00:00:00 to 2024-02-01T00:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias      rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32   Float32   Float32   Float32        Float32        Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     0.200371  0.48092   0.360311      0.105318        0.249065    745
   2 │ Schiermonnikoog Noord    0.233116  0.395703  0.31278       0.164365        0.286398    745
   3 │ L91                      0.209548  0.406436  0.315081      0.107319        0.214189    745
   4 │ wadden eierlandse gat    0.206671  0.401064  0.295731      0.112047        0.215931    745
   5 │ OS11                     0.201116  0.29147   0.233399      0.190277        0.284721    745
   6 │ Europlatform             0.152866  0.276576  0.210192      0.0888641       0.162798    745
   7 │ ijmuiden munitiestort 1  0.237257  0.379518  0.282837      0.131228        0.203483    745
   8 │ K13a                     0.216405  0.405341  0.316175      0.111628        0.211752    745
   9 │ A121                     0.245122  0.586065  0.470653      0.13843         0.307616    745
  10 │ eurogeul E13             0.183078  0.291982  0.222472      0.116545        0.186324    745
  11 │ F3                       0.247691  0.556379  0.432857      0.132675        0.300565    745

- Info: Computing statistics for timespan: training (2021-01-01T00:00:00 to 2023-12-31T23:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias         rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32      Float32   Float32   Float32        Float32        Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     -0.0235253   0.304992  0.220471      0.0387535       0.298615  26257
   2 │ Schiermonnikoog Noord     0.0190984   0.269902  0.201212      0.116793        0.435018  26265
   3 │ L91                      -0.00982977  0.285373  0.2114        0.0570541       0.325363  26260
   4 │ wadden eierlandse gat    -0.00032402  0.263422  0.197254      0.075625        0.363897  26260
   5 │ OS11                      0.0532852   0.233565  0.184213      0.214328        0.616675  26260
   6 │ Europlatform              0.0145963   0.234596  0.180237      0.109784        0.43892   26260
   7 │ ijmuiden munitiestort 1   0.0147016   0.264477  0.199554      0.108689        0.440513  26260
   8 │ K13a                     -0.00327596  0.277686  0.205038      0.0701212       0.353362  26260
   9 │ A121                     -0.0306565   0.360075  0.260417      0.0253638       0.276673  26257
  10 │ eurogeul E13              0.0280379   0.235812  0.182265      0.133733        0.477928  26260
  11 │ F3                       -0.0228695   0.3394    0.24659       0.0326955       0.286453  26257

  ## 10to11_new_lyr1: Same as 10to11 but with modified first layer

- Info: Computing statistics for timespan: test (2024-01-01T00:00:00 to 2024-11-01T00:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias      rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32   Float32   Float32   Float32        Float32        Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     0.183298  0.322587  0.252633       0.25514        0.558552   7016
   2 │ Schiermonnikoog Noord    0.202096  0.298     0.239165       0.361663       0.690158   7016
   3 │ L91                      0.169943  0.291863  0.230584       0.252931       0.531888   7016
   4 │ wadden eierlandse gat    0.145769  0.26229   0.207916       0.245639       0.534543   7016
   5 │ OS11                     0.17647   0.245416  0.199179       0.432207       0.80536    7016
   6 │ Europlatform             0.159473  0.245817  0.197771       0.298598       0.610895   7016
   7 │ ijmuiden munitiestort 1  0.191417  0.278129  0.223229       0.324953       0.641043   7016
   8 │ K13a                     0.169652  0.286279  0.22404        0.275936       0.645195   7016
   9 │ A121                     0.212779  0.386804  0.301817       0.260482       0.54569    7016
  10 │ eurogeul E13             0.157416  0.239944  0.191276       0.318779       0.662561   7016
  11 │ F3                       0.218308  0.365088  0.288783       0.2706         0.548611   7016

- Info: Computing statistics for timespan: 202401 (2024-01-01T00:00:00 to 2024-02-01T00:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias      rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32   Float32   Float32   Float32        Float32        Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     0.295571  0.460831  0.372165       0.154771       0.250243    745
   2 │ Schiermonnikoog Noord    0.313093  0.412947  0.346426       0.219803       0.29714     745
   3 │ L91                      0.277043  0.394008  0.324957       0.14815        0.212913    745
   4 │ wadden eierlandse gat    0.234863  0.349259  0.284633       0.137382       0.208107    745
   5 │ OS11                     0.192763  0.297219  0.231374       0.179405       0.26411     745
   6 │ Europlatform             0.178034  0.291937  0.235005       0.109678       0.178008    745
   7 │ ijmuiden munitiestort 1  0.274732  0.375266  0.298927       0.163113       0.212102    745
   8 │ K13a                     0.242316  0.369392  0.300395       0.132168       0.197078    745
   9 │ A121                     0.323972  0.550035  0.449332       0.173961       0.292595    745
  10 │ eurogeul E13             0.176661  0.290299  0.218647       0.117671       0.183755    745
  11 │ F3                       0.330804  0.515987  0.415391       0.172572       0.291768    745

- Info: Computing statistics for timespan: training (2021-01-01T00:00:00 to 2023-12-31T23:00:00)
stats = 11×7 DataFrame
 Row │ station_name             bias         rmse      mae       relative_bias  scatter_index  count
     │ String                   Float32      Float32   Float32   Float32        Float32        Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────
   1 │ F161                     0.0128715    0.305537  0.226938      0.0801301       0.336584  26257
   2 │ Schiermonnikoog Noord    0.0330409    0.271047  0.202929      0.11872         0.411116  26265
   3 │ L91                      0.0240992    0.280478  0.210631      0.0892391       0.341777  26260
   4 │ wadden eierlandse gat    0.006795     0.254145  0.191799      0.080451        0.352682  26260
   5 │ OS11                     0.00464186   0.209677  0.159203      0.105129        0.441035  26260
   6 │ Europlatform             0.00611882   0.230069  0.174904      0.088274        0.391774  26260
   7 │ ijmuiden munitiestort 1  0.0239152    0.258589  0.19487       0.106237        0.408793  26260
   8 │ K13a                     0.010356     0.276956  0.206591      0.0881879       0.365732  26260
   9 │ A121                     0.0265043    0.356084  0.266592      0.0948812       0.344776  26257
  10 │ eurogeul E13             0.000635562  0.225264  0.170183      0.0876381       0.403958  26260
  11 │ F3                       0.0296908    0.33504   0.251898      0.0962494       0.34721   26257

 Row │ timespan  avg_bias   avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations 
     │ String    Float32    Float32   Float32   Float32            Float32            Int64     
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │ test      0.180602   0.292929  0.232399          0.299721            0.615863         11
   2 │ 202401    0.258168   0.391562  0.316114          0.155334            0.235256         11
   3 │ training  0.0162427  0.27299   0.20514           0.0941033           0.376858         11

   exp
    Row │ timespan  avg_bias     avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32      Float32   Float32   Float32            Float32            Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.153735    0.299339  0.225824          0.224044            0.487743         11
   2 │ 202401     0.286762    0.455056  0.33147           0.162833            0.246546         11
   3 │ training  -0.00532974  0.242402  0.180369          0.0461131           0.298793         11

- exp n_input_channels=32
    Row │ timespan  avg_bias     avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32      Float32   Float32   Float32            Float32            Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.164166    0.299433  0.229507           0.255016           0.526534         11
   2 │ 202401     0.242516    0.403675  0.309315           0.151314           0.241865         11
   3 │ training  -0.00257456  0.23588   0.177073           0.058294           0.309543         11

   - input_noise 0.3
    Row │ timespan  avg_bias    avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32     Float32   Float32   Float32            Float32            Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.0946326  0.278715  0.208254          0.12764             0.376823         11
   2 │ 202401     0.22684    0.409077  0.303179          0.127445            0.222244         11
   3 │ training  -0.0700809  0.265356  0.193004         -0.0358055           0.262117         11
"wave_model_10to11_new_lyr1_3stations_4lyr_3yr_256batch_0p00005reg_16lags_100epochs/average_statis

## 10to11_explyr2: exp layer with 64 channels and 0.3 input noise

 Row │ timespan  avg_bias    avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32     Float32   Float32   Float32            Float32            Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.106421   0.284208  0.211607          0.154213            0.41639          11
   2 │ 202401     0.220587   0.404615  0.304123          0.127308            0.221867         11
   3 │ training  -0.0554233  0.257619  0.186416         -0.0146336           0.268017         11

## 10to11_explyr3: exp layer with minibatch 256->512
- 0.3 input noise, 64 channels

## 10to11_don 
- deep-o-net version
- channels 64,64,64,32
- 100 epochs

 Row │ timespan  avg_bias    avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32     Float32   Float32   Float32            Float32            Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.134349   0.288803  0.218141         0.189942             0.430802         11
   2 │ 202401     0.224912   0.395285  0.299748         0.133685             0.217437         11
   3 │ training  -0.0377701  0.234339  0.169435         0.00705908           0.264404         11

## 10to11_don2
- 32,32,32,16 channels
- 50 epochs

 Row │ timespan  avg_bias    avg_rmse  avg_mae   avg_relative_bias  avg_scatter_index  nstations
     │ String    Float32     Float32   Float32   Float32            Float32            Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │ test       0.133581   0.283976  0.215228          0.191946            0.440453         11
   2 │ 202401     0.248827   0.392806  0.300219          0.150916            0.227693         11
   3 │ training  -0.0319097  0.237224  0.17309           0.0140242           0.271658         11