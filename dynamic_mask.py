            if args.flow_as_mask:
                tracks_primary = track_infos['tracks_l'][
                    :, : args.sequence_length - args.atten_goal + args.pred_num - 1, :
                ]  # .cuda() 已在外面
                B, P, HW, C = tracks_primary.shape
                H = W = int(HW ** 0.5)

                # 1. reshape + permute -> (B*P, 2, H, W)
                tp = tracks_primary.reshape(B * P, H, W, C).permute(0, 3, 1, 2)  # (B*P, 2, H, W)

                # 2. 2×2 平均池化 -> (B*P, 2, H//2, W//2)
                pooled = F.avg_pool2d(tp, kernel_size=2, stride=2)

                # 3. 计算范数 -> (B*P, H//2, W//2)
                norm = torch.norm(pooled, dim=1)  # 沿 channel=2 维度

                # 4. 阈值 + unsqueeze -> 二值 mask (float) (B*P, 1, H//2, W//2)
                threshold = 1.0
                mask = (norm > threshold).unsqueeze(1).float()

                # 5. 膨胀掩模（模拟上下左右扩张）-> (B*P, 1, H//2, W//2)
                dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

                # 6. flatten -> (B*P, 1, L) with L=(H//2)*(W//2)
                h2, w2 = pooled.shape[2], pooled.shape[3]
                mask_reshape = dilated.reshape(B * P, 1, h2 * w2, 1) # 直接就是最终的 primary mask

                # ————————————— 同理处理 gripper track —————————————

                tracks_wrist = track_infos['tracks_gripper_l'][
                    :, : args.sequence_length - args.atten_goal + args.pred_num - 1, :
                ]
                # reshape + permute
                tw = tracks_wrist.reshape(B * P, H, W, C).permute(0, 3, 1, 2)  # (B*P, 2, H, W)

                # 2×2 池化 + 范数 + 阈值
                pooled_w = F.avg_pool2d(tw, kernel_size=2, stride=2)
                norm_w   = torch.norm(pooled_w, dim=1)
                mask_w   = (norm_w > threshold).unsqueeze(1).float()

                # （可选不做膨胀，或同样做 dilate）
                # dilated_w = F.max_pool2d(mask_w, kernel_size=3, stride=1, padding=1)

                # flatten -> (B*P, 1, L)
                wrist_masks_reshape = mask_w.reshape(B * P, 1, h2 * w2, 1)
            
            
            for i in range (40):
                img = label_image_primary.unsqueeze(1).detach().cpu()
                img = unpatchify(img)[i][0].permute(1, 2, 0).detach().cpu().numpy()
                mask_img_np = (img - img.min()) / (img.max() - img.min()) * 255
                mask_img_np = mask_img_np.astype(np.uint8)

                # # 使用 PIL 保存图像
                image = Image.fromarray(mask_img_np)
                image.save(f"./png/image_{i}.png")
                
            for i in range (40):
                mask_img = label_image_primary.detach().cpu()* mask_reshape
                mask_img = unpatchify(mask_img)[i][0].permute(1, 2, 0).detach().cpu().numpy()
                mask_img_np = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min()) * 255
                mask_img_np = mask_img_np.astype(np.uint8)

                # # 使用 PIL 保存图像
                image = Image.fromarray(mask_img_np)
                image.save(f"./png/image_mask_{i}.png")