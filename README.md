# paul_d_growing_minibatches
Code for growing mini-batches

Using Windows? - uncomment lines 21 and 22 

Requirements: use pip to install pytorch_gan_metrics

Based on code from https://github.com/Chavdarova/SVRE/tree/master/svre

How to run: 

main.py --total_step 262500 --dataset cifar10 --adv_loss hinge --sample_step 1000 --data_dir cifar-10-batches-py/ --avg_start -1 --lr_scheduler -1 --g_lr 0.0002 --d_lr 0.0002 --g_beta1 0.5 --d_beta1 0.5 --batch_size 64 --extra False --optim adam --srfb True --eval_step 5000 --batch_s_doubler 25000

This is an example for running 640 epochs of the growing mini-batches variant using growing step-sizes after 32 epochs - this corresponds to 500,000 iterations of the fixed size variant, thus total-step = 262,500. Use the batch_s_doubler argument to indicate at which steps the batch size must double - take into account that doubling the batch size at 25,000 steps (32 epochs) this means that, if you wanted to double the batch size after for instance 64 more epochs (so after 96 epochs covered in total), you need to enter in the step value corresponding to how many iterations this would originally - IE: 96 epochs with a batch size of 64 occurs after 75,000 steps, 96 epochs with an initial batch size of 64 for 25,000 steps (32 epochs) followed by a batch size of 128 is at 50,000 steps. Put very simply -> 25,000 steps @ 64 samples per batch = 32 epochs, 25,000 steps @ 128 samples per batch = 64 epochs, 50,000 steps @ 64 samples per batch = 64 epochs. And so this is why total step should be 262,500 : (25,000 * 64 + 237,500 * 128) / 640 = (500,000 * 64 ) / 640.

DSRI ($ corresponds to shell commands, oc is required): 

$ oc login https://api.dsri2.unimaas.nl:6443 -u <username>

$ oc rsh (name of pod)

$ nohup python main.py --total_step ... [see command above, don't forget the ampersand at the end!] &

[when done:]
$ tar -czf results.tar.gz results nohup.out
