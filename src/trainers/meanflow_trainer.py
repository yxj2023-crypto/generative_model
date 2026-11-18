import torch 
from torch.func import jvp
from tqdm import tqdm
# mean flow 训练流程
# ODE正方向定义为t:0->1,从数据x0到噪声x1,时间越大越靠近噪声。
# 训练：
# 1、从噪声中采样x1，并用log步长采样出t和r，得到xt=t*x1+(1-t)*x0
# 2、用(xt,t,r)经过模型u(x_t,t,r)得到平均速度u_pred(x_t,t,r)
# 3、计算L_meanflow=MSE(u_pred(x_t,t,r) , target) where target=stop_gradient(x1-x0 - (t-r)*JVP_u(x_t,t,r)(x1-x0,1,0))
# 如果是iMF,则需计算v_pred(x_t,t),<这里去网络的另一个输出头>,并计算L_iMF=MSE(u,x1-x0) where u=u_pred(x_t,t,r)+(t-r)*stop_gradient(JVP_u(x_t,t,r)(v_pred(x_t,t),1,0))
# 因为v头也需要监督信号，所以iMF损失中要加上原来的FM_loss=MSE(v_pred(x_t,t),x1-x0)
# 多步生成：噪声到数据，沿ODE反方向--- t>r,x_r=x_t - (t-r)*u+pred(x_t,t,r)
# 单步生成：x0=x1-u(x1,1,0)

# imf中cfg的重设计是另一创新点
# 这里将cfg引导参数作为模型的输入-->输出[u_pred(xt,t,r,c,w),v_pred(xt,t,c,w)]
# 其中c为条件，w为引导强度
# 训练时，对样本进行指派。w固定在[w_min,w_max]的区间
# 先按照ratio_sample指派随机采样的样本，再按照ratio_drop指派丢弃标签的样本
# 第一次采样时，将ratio_samlpe%的样本的w进行随机采样，其余固定为w_min
# 第二次采样时，将ratio_drop%的样本设置为无条件，采样强度固定为w_min
# 两次采样不重复
# 对于指派为有条件的样本，MSE中的目标速度v设置为v_cfg=x1-x0 + (1-1/w)*(v_cond - v_uncond) 这里的v是预测出的速度，不是平均速度
# 测试时可以直接输入目标cfg
from .flow_trainer import FlowMatching
def stop_gradient(x):
    return x.detach()
def adaptive_l2_loss(x,y, gamma=0.0, c=1e-2):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    error=x-y
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stop_gradient(w) * loss).mean()
class MeanFlow(FlowMatching):
    def __init__(self,model,dataloader,optimizer,scheduler,
                 num_epochs=100,eval_per_epoch=20,save_per_epoch=10,num_classes=10,image_size=(28,28),
                 use_cfg=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 raw_model=None,adaptive_l2_gamma=0.0,adaptive_l2_c=1e-2):
        super().__init__(model,dataloader,optimizer,scheduler,
                 num_epochs,eval_per_epoch,save_per_epoch,num_classes,image_size,use_cfg,device)
        self.time_mean=-0.4
        self.time_std=1.0
        self.cfg_scale_range=[1.0,7.5]
        self.cfg_ratio_sample=0.75
        self.cfg_ratio_drop=0.1
        # Optional unwrapped model for auxiliary predictions and sampling.
        self.raw_model=model if raw_model is None else raw_model
        self.adaptive_l2_gamma=adaptive_l2_gamma
        self.adaptive_l2_c=adaptive_l2_c

    def adaptive_loss(self,x,y):
        return adaptive_l2_loss(x,y,self.adaptive_l2_gamma,self.adaptive_l2_c)
    @staticmethod
    def normalize(x):
        return x*2-1
    @staticmethod
    def denormalize(x):
        return ((x+1)/2).clamp(0,1)
    def sampling(self,x0):
        # x1(B,C,H,W)
        x0=x0.to(self.device)
        B=x0.shape[0]
        # t:(B,)每个批次一个
        time_sample=(torch.randn(B,2,device=self.device)*self.time_std+self.time_mean).sigmoid()
        t=torch.max(time_sample,dim=-1).values
        r=torch.min(time_sample,dim=-1).values

        x1=torch.randn_like(x0).to(self.device)
        t_view=t[:,None,None,None]
        xt=t_view*x1+(1-t_view)*x0
        return xt,t,r,x1
    def sampling_cfg(self,y):
        y=y.clone()
        B=y.shape[0]
        low,high=self.cfg_scale_range
        w = torch.ones(B,device=self.device)
        # 指派条件采样
        sample_num = int(B*self.cfg_ratio_sample)
        indices=torch.randperm(B,device=self.device)[:sample_num]
        w[indices] = torch.empty(sample_num, device=y.device).uniform_(low, high)
        # 指派条件丢弃
        drop_mask = torch.rand(B,device=self.device) < self.cfg_ratio_drop 
        y[drop_mask] = self.num_classes#将类别数目作为无条件的编码，对应模型中0~num_classes-1个类别和num_classes的无条件
        w[drop_mask] = low
        return y,w
    def meanflow_loss(self,v,xt,u_pred,v_pred,t,r):
        dt=(t-r)[:,None,None,None]
        def u_fn(x,t,r):
            return self.raw_model(x,t,r)[0]
        FM_loss=self.adaptive_loss(v_pred,v)
        with torch.no_grad():
            _,dudt=jvp(u_fn,(xt,t,r),(v,torch.ones_like(t),torch.zeros_like(r)))
        target=stop_gradient(v-dt*dudt)
        return self.adaptive_loss(u_pred,target)+FM_loss
    def imeanflow_loss(self,v,xt,u_pred,v_pred,t,r,y=None,w=None):
        dt=(t-r)[:,None,None,None]
        def u_fn(x,t,r):
            if y is None:
                return self.raw_model(x,t,r)[0]
            return self.raw_model(x,t,r,y,w)[0]
        fm_loss=self.adaptive_loss(v_pred,v)
        with torch.no_grad():
            _,dudt=jvp(u_fn,(xt,t,r),(stop_gradient(v_pred),torch.ones_like(t),torch.zeros_like(r)))
        target=stop_gradient(v-dt*dudt)
        return self.adaptive_loss(u_pred,target)+fm_loss
    @torch.no_grad()
    def generate(self,num_step,shape,y=None,cfg_scale=2.0):
        times=torch.linspace(1.0,0.0,num_step+1).to(self.device)
        self.model.eval()
        xt=torch.randn(shape,device=self.device)
        if not self.use_cfg:
            y_input=None
            w=None
        else:
            if y is not None:
                w=torch.ones(xt.shape[0],device=xt.device)*cfg_scale
                y_input=torch.as_tensor(y,device=xt.device,dtype=torch.long).reshape(-1)
            else:
                y_uncond=torch.ones(xt.shape[0],device=xt.device)*self.num_classes
                y_input=y_uncond.to(torch.long)
                w=torch.ones(xt.shape[0],device=xt.device)
        for i in range(num_step):
            t=times[i].expand(shape[0])
            r=times[i+1].expand(shape[0])
            u=self.raw_model(xt,t,r,y_input,w)[0]
            xr=xt-(t-r)[:,None,None,None]*u
            xt=xr 
        return self.denormalize(xt)
    
    def train_epoch(self):
        self.model.train()
        total_loss=0.0
        total_samples=0
        for batch in tqdm(self.dataloader):
            self.optimizer.zero_grad(set_to_none=True)
            x0=self.normalize(x0.to(self.device))
            xt,t,r,x1=self.sampling(x0)
            v_target=(x1-x0).detach()
            y_cfg,w=None,None
            if self.use_cfg:
                y_cfg,w=self.sampling_cfg(y.to(self.device).reshape(-1))
                mask=(y_cfg != self.num_classes) & (w != 1)
                if mask.any():
                    with torch.no_grad():
                        xt_m=xt[mask]
                        t_m,r_m=t[mask],r[mask]
                        y_m,w_m=y_cfg[mask],w[mask]
                        _,v_cond=self.raw_model(xt_m,t_m,r_m,y_m,w_m)
                        _,v_uncond=self.raw_model(
                            xt_m,t_m,r_m,self.num_classes*torch.ones_like(y_m),torch.ones_like(w_m))
                        factor=(1-1/w_m)[:,None,None,None]
                        v_target[mask] += factor*(v_cond-v_uncond)
            u_pred,v_pred=self.model(xt,t,r,y_cfg,w)
            loss=self.imeanflow_loss(v_target,xt,u_pred,v_pred,t,r,y_cfg,w)
            loss.backward()
            self.optimizer.step()
            batch_size=batch[0].shape[0]
            total_loss+=loss.detach().item()*batch_size
            total_samples+=batch_size
            if self.scheduler is not None:
                self.scheduler.step()
        if total_samples == 0:
            raise ValueError("dataloader contains no samples")
        return total_loss/total_samples
