# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Timing Summary
Size: 64
    fast: 0.00393
    gpu: 0.00618
Size: 128
    fast: 0.01637
    gpu: 0.01444
Size: 256
    fast: 0.10254
    gpu: 0.04974
Size: 512
    fast: 0.99804
    gpu: 0.19792
Size: 1024
    fast: 10.46135
    gpu: 0.84202

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Average time per epoch: 2.3051241 seconds
Epoch  0  loss  6.9587422982900975 correct 16
Epoch  10  loss  6.594902150832109 correct 26
Epoch  20  loss  5.756094313374342 correct 26
Epoch  30  loss  6.166413923877062 correct 26
Epoch  40  loss  4.487460045852151 correct 26
Epoch  50  loss  6.487962998943811 correct 26
Epoch  60  loss  5.545720591152609 correct 26
Epoch  70  loss  4.865006732855122 correct 26
Epoch  80  loss  4.586528399377377 correct 26
Epoch  90  loss  4.415450956971519 correct 27
Epoch  100  loss  4.612791994407601 correct 28
Epoch  110  loss  4.338148087232733 correct 28
Epoch  120  loss  3.829411484800172 correct 31
Epoch  130  loss  3.8599094402722547 correct 32
Epoch  140  loss  4.356045299887103 correct 34
Epoch  150  loss  3.2120745502529093 correct 37
Epoch  160  loss  2.8070111517693146 correct 38
Epoch  170  loss  2.9007301585602847 correct 40
Epoch  180  loss  2.820521117753598 correct 42
Epoch  190  loss  2.262376182138234 correct 43
Epoch  200  loss  2.8042861087316906 correct 43
Epoch  210  loss  2.288741373384258 correct 42
Epoch  220  loss  1.9923873878180394 correct 44
Epoch  230  loss  1.8776817483087098 correct 43
Epoch  240  loss  2.018148534787861 correct 43
Epoch  250  loss  3.2742914344209932 correct 43
Epoch  260  loss  1.306553377306564 correct 44
Epoch  270  loss  2.228918362407749 correct 45
Epoch  280  loss  1.6229950708910423 correct 44
Epoch  290  loss  1.492796714313593 correct 44
Epoch  300  loss  1.079418906646854 correct 44
Epoch  310  loss  1.931929240310635 correct 45
Epoch  320  loss  2.7629629753276808 correct 45
Epoch  330  loss  2.2110100160920534 correct 45
Epoch  340  loss  0.9312070020177281 correct 45
Epoch  350  loss  0.6551966943016229 correct 44
Epoch  360  loss  0.2521247984544978 correct 44
Epoch  370  loss  1.0348470648760957 correct 44
Epoch  380  loss  2.5967148929801751 correct 45
Epoch  390  loss  0.1752934227768601 correct 46
Epoch  400  loss  1.3387690917522024 correct 46
Epoch  410  loss  0.12349065713692325 correct 46
Epoch  420  loss  1.1827659180663745 correct 45
Epoch  430  loss  0.407015233427827 correct 47
Epoch  440  loss  0.9444224507965846 correct 47
Epoch  450  loss  1.9782978293150588 correct 45
Epoch  460  loss  0.6960157801852501 correct 48
Epoch  470  loss  0.3468619486754105 correct 50
Epoch  480  loss  0.464574367407297 correct 49
Epoch  490  loss  0.31196723807736065 correct 50

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Average time per epoch: 2.7632247 seconds
Epoch  0  loss  6.275946371952103 correct 29
Epoch  10  loss  6.427395389137238 correct 26
Epoch  20  loss  5.746495417440606 correct 26
Epoch  30  loss  4.527225207365584 correct 37
Epoch  40  loss  6.312847884005965 correct 40
Epoch  50  loss  4.530964753915853 correct 45
Epoch  60  loss  3.9802471093080887 correct 42
Epoch  70  loss  4.235252476982022 correct 44
Epoch  80  loss  4.071770774972001 correct 45
Epoch  90  loss  2.9865296229802176 correct 46
Epoch  100  loss  2.759529356659579 correct 45
Epoch  110  loss  3.120433979536339 correct 45
Epoch  120  loss  2.1811675523679144 correct 45
Epoch  130  loss  3.2126550617458305 correct 46
Epoch  140  loss  2.8819605131336656 correct 45
Epoch  150  loss  1.6361299431721101 correct 46
Epoch  160  loss  2.266156366522865 correct 46
Epoch  170  loss  1.8866843219362588 correct 47
Epoch  180  loss  3.105543846589744 correct 44
Epoch  190  loss  1.6803154340519058 correct 47
Epoch  200  loss  2.349993480483264 correct 46
Epoch  210  loss  1.3067051484024714 correct 47
Epoch  220  loss  1.7487203473165276 correct 48
Epoch  230  loss  0.38815031119518667 correct 47
Epoch  240  loss  2.7660941441043048 correct 48
Epoch  250  loss  2.4751840107275527 correct 47
Epoch  260  loss  0.7110338743746001 correct 47
Epoch  270  loss  1.7294706405487128 correct 47
Epoch  280  loss  1.2406014447247333 correct 47
Epoch  290  loss  2.4829678989033326 correct 47
Epoch  300  loss  2.1418192421231383 correct 47
Epoch  310  loss  2.1699342385017837 correct 48
Epoch  320  loss  0.5539615349537844 correct 47
Epoch  330  loss  1.2597196529541177 correct 49
Epoch  340  loss  1.713423818479836 correct 47
Epoch  350  loss  1.2910145326829574 correct 47
Epoch  360  loss  0.8418493964293972 correct 47
Epoch  370  loss  1.888855237371782 correct 46
Epoch  380  loss  0.7433540964054441 correct 48
Epoch  390  loss  1.295779701750994 correct 47
Epoch  400  loss  1.230750045798457 correct 49
Epoch  410  loss  0.7958429923454933 correct 50
Epoch  420  loss  0.45745615294336717 correct 48
Epoch  430  loss  1.6861787015338754 correct 49
Epoch  440  loss  1.1370339334130948 correct 50
Epoch  450  loss  0.3888811141629669 correct 50
Epoch  460  loss  1.3049701714514663 correct 49
Epoch  470  loss  0.31925883613325867 correct 50
Epoch  480  loss  0.7339960549125752 correct 49
Epoch  490  loss 0.584426816922056 correct 50

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Average time per epoch: 2.1236243 seconds
Epoch  0  loss  7.1657125903183925  correct 29
Epoch  10  loss  4.55016188401549  correct 36
Epoch  20  loss  3.6455936659593835  correct 42
Epoch  30  loss  0.9813818584230645  correct 43
Epoch  40  loss  1.153174233444056  correct 47
Epoch  50  loss  1.9617677457836218  correct 47
Epoch  60  loss  0.9904842235130531  correct 48
Epoch  70  loss  0.7509976847330667  correct 47
Epoch  80  loss  1.728870608180012  correct 49
Epoch  90  loss  0.8352726335715991  correct 48
Epoch  100  loss  1.3902978809152449  correct 48
Epoch  110  loss  0.7470444814018761  correct 48
Epoch  120  loss  0.8612947781963672  correct 49
Epoch  130  loss  1.1167308638057278  correct 50
Epoch  140  loss  0.5250958079678305  correct 49
Epoch  150  loss  0.3552522227874904  correct 50

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Average time per epoch: 0.0692421 seconds
Epoch  0  loss  4.078085632227521  correct 44
Epoch  10  loss  1.3673567447379769  correct 48
Epoch  20  loss  0.6243446756326712  correct 50
Epoch  30  loss  0.30233155387698885  correct 50
Epoch  40  loss  1.0260046379940806  correct 50
Epoch  50  loss  0.7347511034432715  correct 50
Epoch  60  loss  0.34305235219006597  correct 50
Epoch  70  loss  0.3314165750127725  correct 50

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Average time per epoch: 0.0735026 seconds
Epoch  0  loss  6.706463343863391  correct 31
Epoch  10  loss  4.350852972363642  correct 45
Epoch  20  loss  2.8594061021537267  correct 42
Epoch  30  loss  3.160059919637691  correct 46
Epoch  40  loss  2.432313977720815  correct 47
Epoch  50  loss  2.769658095887044  correct 48
Epoch  60  loss  2.1220231965053684  correct 46
Epoch  70  loss  1.012762228866  correct 44
Epoch  80  loss  2.527577681105108  correct 49
Epoch  90  loss  2.0165206938263625  correct 49
Epoch  100  loss  3.080745914082142  correct 50
Epoch  110  loss  1.0339323273966774  correct 49
Epoch  120  loss  0.545320574648509  correct 50
Epoch  130  loss  1.6790350306428827  correct 50
Epoch  140  loss  1.2663860883493303  correct 50
Epoch  150  loss  1.4750307291003906  correct 49
Epoch  160  loss  1.160310615750876  correct 48
Epoch  170  loss  1.2384864441858554  correct 49
Epoch  180  loss  0.7982019156209521  correct 50
Epoch  190  loss  0.8538953897895267  correct 50
Epoch  200  loss  0.45193870145319665  correct 50

# !cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Average time per epoch: 0.07854930 seconds
Epoch  0  loss  6.422365927365998  correct 31
Epoch  10  loss  6.8288923296099675  correct 33
Epoch  20  loss  3.8660706122100135  correct 44
Epoch  30  loss  4.5706389085800865  correct 45
Epoch  40  loss  2.451804759947005  correct 43
Epoch  50  loss  2.931681128506071  correct 48
Epoch  60  loss  1.0089524902120945  correct 48
Epoch  70  loss  1.5497790545789598  correct 49
Epoch  80  loss  0.7837609489315328  correct 49
Epoch  90  loss  1.2042550609999163  correct 50
Epoch  100  loss  1.4633700996670678  correct 50
Epoch  110  loss  0.9959195692329276  correct 49
Epoch  120  loss  0.6531943682503615  correct 49
Epoch  130  loss  0.630325342346656  correct 50
Epoch  140  loss  1.1880419307951915  correct 50
Epoch  150  loss  0.7869846189519676  correct 49
Epoch  160  loss  0.9694733086940014  correct 50
Epoch  170  loss  1.1408008963457195  correct 50
Epoch  180  loss  0.6304221396984481  correct 50
Epoch  190  loss  0.5020505038668658  correct 49
Epoch  200  loss  0.8977571537286655  correct 50
