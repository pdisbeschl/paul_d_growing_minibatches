main.py \
	--dataset ${dataset} --adv_loss hinge \
	--sample_step 1000 \
	--data_dir ${data_dir} \
	--g_lr ${g_lr} \
	--d_lr ${d_lr} \
	--avg_start ${avg_start} \
    --lr_scheduler ${gamma} \
	--batch_size ${bsize} \
	--extra True --optim sgd \
	--svrg
