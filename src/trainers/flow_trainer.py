import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
# flow（1-reflow）的训练流程
# 1、从噪声分布采样x0，数据分布采样x1,时间步t
# 2、加噪产生xt=t*x1+(1-t)*x0，(xt,t)->model->u_t(当前时刻的速度)
# 3、损失计算:loss=MSE(u_t,x1-x0)
# flow的测试流程
# 1、从噪声分布采样x0，获取训练好的模型model
# ->生成参数：N步，步长h=1/N
# 2、欧拉法求解微分方程：xt_=xt+h*model(xt,t),xt=xt_,t=t+h
# 3、t=1输出结果

class FlowMatching:
    def __init__(self,model,dataloader,optimizer,scheduler,
                 num_epochs=100,eval_per_epoch=20,save_per_epoch=10,num_classes=10,image_size=(28,28),
                 use_cfg=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model=model.to(device)
        self.device=device 
        self.dataloader=dataloader 
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.use_cfg=use_cfg
        self.num_epochs=num_epochs
        self.eval_per_epoch=eval_per_epoch
        self.save_per_epoch=save_per_epoch
        self.num_classes=num_classes
        self.image_size=image_size
    def sampling(self,x1):
        # x1(B,C,H,W)
        x1=x1.to(self.device)
        B=x1.shape[0]
        # t:(B,)每个批次一个
        t=torch.rand(B).to(self.device)
        x0=torch.randn_like(x1).to(self.device)
        t_view=t[:,None,None,None]
        xt=t_view*x1+(1-t_view)*x0
        return xt,t,x0
    def generate(self,num_step,shape,y=None,cfg_scale=5.0):
        """y=None 为无条件；否则以 CFG 组合条件与无条件速度。"""
        h=1/num_step
        if y is not None:
            if not self.use_cfg:
                raise ValueError('Conditional generation requires CFG training')
            y=torch.as_tensor(y,device=self.device,dtype=torch.long).reshape(-1)
        self.model.eval()
        with torch.no_grad():
            xt=torch.randn(shape,device=self.device)
            for step in range(num_step):
                t=torch.full((shape[0],),step*h,device=self.device)
                u_uncond=self.model(xt,t)
                if y is None:
                    u_pred=u_uncond
                else:
                    u_cond=self.model(xt,t,y=y)
                    u_pred=u_uncond+cfg_scale*(u_cond-u_uncond)
                xt=xt+h*u_pred
            return xt
        
    def loss(self,x1,x0,u_pred):
        return F.mse_loss(u_pred,x1-x0)
    def train_epoch(self):
        """训练一轮，CFG 模式对齐原项目的双份 batch 实现。"""
        self.model.train()
        total_loss=0.0
        total_samples=0
        for batch in tqdm(self.dataloader):
            # MNIST 返回 (图片, 标签)；无条件训练只使用图片。
            x1,y=batch
            x1=x1.to(self.device)
            xt,t,x0=self.sampling(x1)

            self.optimizer.zero_grad()
            if self.use_cfg:
                y=y.to(self.device)
                y=y.reshape(-1)#(B,)

                # cfg：一份输入拆两份，一份无条件（y置为-1），一份有条件
                xt=torch.cat([xt,xt],dim=0)
                t=torch.cat([t,t],dim=0)
                y=torch.cat([y,-torch.ones_like(y)],dim=0)
                x1=torch.cat([x1,x1],dim=0)
                x0=torch.cat([x0,x0],dim=0)
                u_pred=self.model(xt,t,y=y)
            else:
                u_pred=self.model(xt,t)
            loss=self.loss(x1,x0,u_pred)
            loss.backward()
            self.optimizer.step()

            batch_size=x1.shape[0]
            total_loss+=loss.detach().item()*batch_size
            total_samples+=batch_size

        if total_samples == 0:
            raise ValueError("dataloader contains no samples")
        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss/total_samples
    def train(self,result_dir,generate_num_step=50,num_samples=100,cfg_scale=5.0):
        result_dir=Path(result_dir)
        (result_dir/'checkpoints').mkdir(parents=True,exist_ok=True)
        (result_dir/'samples').mkdir(parents=True,exist_ok=True)
        # 读取图片形状；每次评估生成 num_samples 张 MNIST 图片。
        images,_=next(iter(self.dataloader))
        shape=(num_samples,*images.shape[1:])
        y=torch.arange(num_samples,device=self.device)%self.num_classes
        loss_history=[]
        for epoch in range(self.num_epochs):
            loss=self.train_epoch()
            loss_history.append(loss)
            (result_dir/'loss_history.json').write_text(json.dumps(loss_history),encoding='utf-8')
            epoch_num=epoch+1
            print(f'Epoch {epoch_num}/{self.num_epochs}: loss={loss:.6f}')
            if epoch_num % self.save_per_epoch==0 or epoch_num==self.num_epochs:
                torch.save({
                    'model':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'scheduler':self.scheduler.state_dict() if self.scheduler is not None else None,
                    'epoch':epoch_num,
                    'loss':loss,
                    'use_cfg':self.use_cfg,
                },result_dir/'checkpoints'/f'epoch_{epoch_num}.pth')
            if epoch_num % self.eval_per_epoch==0 or epoch_num==self.num_epochs:
                samples=self.generate(generate_num_step,shape)
                save_image(samples.clamp(0,1),result_dir/'samples'/f'epoch_{epoch_num}_uncond.png',nrow=10)
                if self.use_cfg:
                    samples=self.generate(generate_num_step,shape,y=y,cfg_scale=cfg_scale)
                    save_image(samples.clamp(0,1),result_dir/'samples'/f'epoch_{epoch_num}_cfg.png',nrow=10)
            
