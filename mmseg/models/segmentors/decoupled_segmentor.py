import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy, copy
import mmcv

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from ..builder import build_discriminator
'''

DecoupledSegmentor
one more encoder compared with the original encoder.
one source encoder and one target encoder
can be switched in the training process

'''


@SEGMENTORS.register_module()
class DecoupledSegmentor(BaseSegmentor):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 discriminator1_0=None,
                 discriminator1_1=None,
                 discriminator1_2=None, 
                 discriminator1_3=None,
                 discriminator2=None):
        super(DecoupledSegmentor, self).__init__(init_cfg)
        self.source_label = 0
        self.target_label = 1
        self.bce_loss = nn.BCEWithLogitsLoss()
        if pretrained is not None:
            # assert backbone.get('pretrained') is None, \
            #    'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

        # mmcv.print_log("Building DecoupledSegmentor")
        # mmcv.print_log("Building Backbone:")
        # mmcv.print_log(backbone)
        self.source_encoder = builder.build_backbone(backbone)
        self.target_encoder = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.model_D1_0 = build_discriminator(discriminator1_0)
        # self.model_D1_1 = build_discriminator(discriminator1_1)
        # self.model_D1_2 = build_discriminator(discriminator1_2)
        # self.model_D1_3 = build_discriminator(discriminator1_3)
        self.model_D2 = build_discriminator(discriminator2)
        # mmcv.print_log("Successfully build discriminator\n")
        assert self.with_decode_head



        
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes


    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)



    def extract_feat(self, img):
        return None

    def source_extract_feat(self, img):
        """extract source_feature
           using source encoder and img 
        """
        x = self.source_encoder(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    

    def target_extract_feat(self, img):
        """extract target_feature
            using target encoder and img
        """

        x = self.target_encoder(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def encode_decode(self, img):
        return None

    def source_encode_decode(self, img, img_metas):
        """Encode images with source encoder and decode into a semantic segmentation
        map of the same size as input."""
        x = self.source_extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    def target_encode_decode(self, img, img_metas):
        """Encode images with target encoder and decode into a semantic segmentation
        map of the same size as input."""
        # mmcv.print_log("val using target encoder\n")
        x = self.target_extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    


    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses
    

    def encode_forward_dummy(self, img):
        """Dummy encoder forward function."""
        seg_logit = self.source_encode_decode(img, None)
        return seg_logit
    

    def target_forward_dummy(self, img):
        """Dummy decoder forward function."""
        seg_logit = self.target_encode_decode(img, None)
        return seg_logit
    

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      target_encoder=False,
                      target_data=False,
                      mixed_training=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if target_encoder==True and target_data==True and mixed_training==False:
            # target encoder and target data
            # mmcv.print_log("target encoder and target data\n")
            x = self.target_extract_feat(img)
            out = self._decode_head_forward_test(x, img_metas)
            
            for param in self.model_D1_0.parameters():
                param.requires_grad = False
            
            for param in self.model_D2.parameters():
                param.requires_grad = False
            
            D1_out = self.model_D1_0(F.interpolate(x[0], size=[512, 512], mode='bilinear', align_corners=True))
            loss_adv_tgt = 0.1 * self.bce_loss(D1_out, torch.FloatTensor(D1_out.data.size()).fill_(self.source_label).cuda())
            # print(f'loss_adv_tgt_D1_0 {loss_adv_tgt_D1_0}')
            losses['loss_adv_tgt'] = loss_adv_tgt
            

            
            for param in self.model_D1_0.parameters():
                param.requires_grad=True
            

            for param in self.model_D2.parameters():
                param.requires_grad=True

            tgt_D2_pred = self.model_D2(F.interpolate(out.detach(), size=[512, 512], mode='bilinear', align_corners=True))
            loss_D2_tgt = 0.5 * self.bce_loss(tgt_D2_pred, torch.FloatTensor(tgt_D2_pred.data.size()).fill_(self.target_label).cuda())
            losses['loss_D2_tgt'] = loss_D2_tgt
            # print(f"loss D2 tgt {loss_D2_tgt}\n")
            tgt_D1_0_pred = self.model_D1_0(F.interpolate(x[0].detach(), size=[512, 512], mode='bilinear', align_corners=True))
            loss_D1_0_tgt = 0.5 * self.bce_loss(tgt_D1_0_pred, torch.FloatTensor(tgt_D1_0_pred.data.size()).fill_(self.target_label).cuda())
            losses['loss_D1_0_tgt'] = loss_D1_0_tgt
            # print(f"loss D1 0 tgt {loss_D1_0_tgt}\n")
            

            
            
        elif target_encoder==False and target_data==False and mixed_training==False:
            # mmcv.print_log("source encoder and source data\n")
            x = self.source_extract_feat(img)
            if return_feat:
                losses['features'] = x
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight)
            out = self._decode_head_forward_test(x, img_metas)
            losses.update(loss_decode)
            
            for param in self.model_D1_0.parameters():
                param.requires_grad = False

            for param in self.model_D2.parameters():
                param.requires_grad = False
            D2_out = self.model_D2(F.interpolate(out, size=[512, 512], mode='bilinear', align_corners=True))
            loss_adv_src = 0.01 * self.bce_loss(D2_out, torch.FloatTensor(D2_out.data.size()).fill_(self.target_label).cuda())
            # print(f"loss adv src {loss_adv_src}\n")
            losses['loss_adv_src'] = loss_adv_src
            
            
            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg, seg_weight)
                losses.update(loss_aux)
            
            
            for param in self.model_D1_0.parameters():
                param.requires_grad = True

            for param in self.model_D2.parameters():
                param.requires_grad = True
            src_D1_0_pred = self.model_D1_0(F.interpolate(x[0].detach(), size=[512, 512], mode='bilinear', align_corners=True))
            loss_D1_0_src = self.bce_loss(src_D1_0_pred, torch.FloatTensor(src_D1_0_pred.data.size()).fill_(self.source_label).cuda())
            losses['loss_D1_0_src'] = loss_D1_0_src
            


            src_D2_pred = self.model_D2(F.interpolate(out.detach(), size=[512, 512], mode='bilinear', align_corners=True))
            loss_D2_src = self.bce_loss(src_D2_pred, torch.FloatTensor(src_D2_pred.data.size()).fill_(self.source_label).cuda())
            losses['loss_D2_src'] = loss_D2_src
            

        elif target_encoder==True and target_data==False and mixed_training==False:
            # mmcv.print_log("target encoder and source data\n")
            x = self.target_extract_feat(img)
            if return_feat:
                losses['features'] = x
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight)
            out = self._decode_head_forward_test(x, img_metas)
            losses.update(loss_decode)


        else: 
            
            # mmcv.print_log("mix training\n")
            x = self.target_extract_feat(img)
            if return_feat:
                losses['features'] = x
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        seg_weight)
            out = self._decode_head_forward_test(x, img_metas)
            # mmcv.print_log(f'loss decode {loss_decode}')
            losses.update(loss_decode)
            
            for param in self.model_D1_0.parameters():
                param.requires_grad = False
            for param in self.model_D2.parameters():
                param.requires_grad = False

            D1_0_out_mix = self.model_D1_0(F.interpolate(x[0], size=[512, 512], mode='bilinear', align_corners=True))
            loss_adv_tgt_mix_0 = 0.01 * self.bce_loss(D1_0_out_mix, torch.FloatTensor(D1_0_out_mix.data.size()).fill_(self.source_label).cuda())
            losses['loss_adv_tgt_mix_0']=loss_adv_tgt_mix_0
            

            

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(
                    x, img_metas, gt_semantic_seg, seg_weight)
                losses.update(loss_aux)
            
        return losses
    


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, target_encoder=True):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if target_encoder:
                    crop_seg_logit = self.target_encode_decode(crop_img, img_meta)
                else:
                    crop_seg_logit = self.source_encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds



    def whole_inference(self, img, img_meta, rescale, target_encoder=True):
        """Inference with full image."""

        if target_encoder:
            seg_logit = self.target_encode_decode(img, img_meta)
        else:
            seg_logit = self.source_encode_decode(img, img_meta)

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit
    

    def inference(self, img, img_meta, rescale, target_encoder=True):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        # print(f'Inference Target Encoder: {target_encoder}\n')
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, target_encoder)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, target_encoder)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output
    

    def simple_test(self, img, img_meta, rescale=True, target_encoder=True):
        """Simple test with single image."""
        target_encoder=True
        # print(f"Simple Test Target Encoder: {target_encoder}\n")
        
        seg_logit = self.inference(img, img_meta, rescale, target_encoder)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True, target_encoder=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        target_encoder=True
        # print(f"Aug Test Target Encoder: {target_encoder}\n")
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale, target_encoder)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale, target_encoder)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred