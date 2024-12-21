import torch
import numpy as np
import PIL.Image as Image
import random
import json
import cv2


from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from skimage import transform
from PIL import ImageEnhance
import xml.etree.ElementTree as ET

# from tools.tools import torch2png
from torchvision.transforms import Resize, Compose, InterpolationMode
# from depth_anything.util.transform import PrepareForNet

class NormalizeImage(object):
    def __init__(self, scale=255):
        self.scale = scale
    def __call__(self, image):
        image = image / self.scale
        return image

class PIL2Numpy(object):
    def __init__(self):
        pass
    def __call__(self, image):
        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        else:
            image = image[:, :, 0:3]
        return image

class DepthVaild(object):
    def __init__(self, min=300, max=2000):
        self.min = min
        self.max = max
    def __call__(self, depth):
        depth[depth > self.max] = 0
        depth[depth < self.min] = 0
        return depth

class Numpy4Tensor(object):
    def __init__(self):
        pass
    def __call__(self, image):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image).astype(np.float32)
        return image
    
class ResizeNumpy(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = cv2.resize(img, self.size, interpolation=self.interpolation)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            return img
        return img
    
class realMultiFlowDataset(Dataset):
    def __init__(
        self, 
        cfg, 
        dataset_path, 
        img_width, 
        img_height, 
        oral_width, 
        oral_height, 
        HR_img_width=1280,
        HR_img_height=720,
        test_boolen=False, 
        srcs=4,
        tgt=None,
        DAtransform=None,
        min_vaild=300,
        max_vaild=1800
    ):
        # dataset_path to train or test folser
        super().__init__()
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.HR_img_width = HR_img_width
        self.HR_img_height = HR_img_height
        self.oral_width = oral_width
        self.oral_height = oral_height
        self.test_boolen = test_boolen
        self.cfg = cfg
        self.srcs = srcs
        self.tgt = tgt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.DAtransform = DAtransform

        self.scale = 1

        camera_path = Path(self.dataset_path) / Path('file')
        camera_json = camera_path / Path('parameters.json')
        self.camera_json = camera_json
        
        dataset_path = Path(dataset_path)
        if not dataset_path.is_dir():
            raise Exception(f"{str(dataset_path)} does not exist")
        
        data_path = dataset_path / Path('images')
        mask_path = dataset_path / Path('masks')
        
        all_people = []
        people_paths = sorted(data_path.glob('*'))
        for p in people_paths:
            data_paths = sorted(p.glob('*'))
            image_files = []
            depth_files = []
            for i in range(len(data_paths)):
                camera_path = data_paths[i]
                image_paths = sorted(camera_path.glob('*.png'), key=lambda x: int(x.name[4:-4]))
                depth_paths = sorted(camera_path.glob('*.tiff'), key=lambda x: int(x.name[4:-5]))
                image_files.append(image_paths)
                depth_files.append(depth_paths)
            all_files = [image_files, depth_files]
            all_people.append(all_files)
        self.all_people = all_people

        all_mask = []
        mask_paths = sorted(mask_path.glob('*'))
        for m in mask_paths:
            mask_data_paths = sorted(m.glob('*'))
            mask_files = []
            for i in range(len(mask_data_paths)):
                masks = sorted(mask_data_paths[i].glob('*.png'), key=lambda x: int(x.stem))
                mask_files.append(masks)
            all_mask.append(mask_files)
        self.all_mask = all_mask

        self.Ks, self.poses, self.HR_Ks = self.load_camera_xml()

        self.single_people_length = len(self.all_people[0][0][0])

        self.image_exp_transform = Compose([
            Resize((self.img_height, self.img_width), interpolation=InterpolationMode.BILINEAR),
            PIL2Numpy(),
            NormalizeImage(),
            Numpy4Tensor()
        ])
        self.image_HR_transform = Compose([
            Resize((self.HR_img_height, self.HR_img_width), interpolation=InterpolationMode.BILINEAR),
            PIL2Numpy(),
            NormalizeImage(),
            Numpy4Tensor()
        ])
        self.depth_transform = Compose([
            ResizeNumpy((self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST),
            DepthVaild(min_vaild, max_vaild),
            Numpy4Tensor()
        ])
    
    def load_camera_xml(self):
        xml_path = Path(self.dataset_path)
        xml_path = xml_path / Path('parameter.xml')
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        Ks = []
        Ks_HR = []
        poses = []
        for i in range(self.srcs):
            name = f'camera_{i}_intrinsic_matrix'
            K = np.reshape(np.array(root.find(name).find('data').text.split(), dtype=np.float32), [3, 3])
            K[0, 0] = K[0, 0] / self.oral_width * self.img_width
            K[1, 1] = K[1, 1] / self.oral_height * self.img_height
            K[0, 2] = 0.5 * self.img_width
            K[1, 2] = 0.5 * self.img_height
            K_eye = np.eye(4)
            K_eye[0:3, 0:3] = K
            Ks.append(K_eye)

            K = np.reshape(np.array(root.find(name).find('data').text.split(), dtype=np.float32), [3, 3])
            K[0, 0] = K[0, 0] / self.oral_width * self.HR_img_width
            K[1, 1] = K[1, 1] / self.oral_height * self.HR_img_height
            K[0, 2] = 0.5 * self.HR_img_width
            K[1, 2] = 0.5 * self.HR_img_height
            K_eye = np.eye(4)
            K_eye[0:3, 0:3] = K
            Ks_HR.append(K_eye)

            name = f'camera_{i}_rotation_matrix'
            R = np.reshape(np.array(root.find(name).find('data').text.split(), dtype=np.float32), [3, 3])
            name = f'camera_{i}_translation_vector'
            T = np.reshape(np.array(root.find(name).find('data').text.split(), dtype=np.float32), [3, 1])
            pose = np.concatenate((R, T), axis=1)
            eye = np.eye(4)
            eye[0:3] = pose
            pose = eye
            # change mm to m
            pose[:, 3] = pose[:, 3] / self.scale
            poses.append(pose)
        return Ks, poses, Ks_HR
    
    def center_crop_numpy(self, image_array, target_size=1080):
        # image_array: H, W or H, W, C
        height, width = image_array.shape[:2]
        
        if target_size > height or target_size > width:
            raise ValueError("目标尺寸大于图像的高度或宽度。")
            
        start_y = (height - target_size) // 2
        start_x = (width - target_size) // 2
        
        end_y = start_y + target_size
        end_x = start_x + target_size
        
        if image_array.ndim == 3:
            cropped_array = image_array[start_y:end_y, start_x:end_x, :]
        else:
            cropped_array = image_array[start_y:end_y, start_x:end_x]
        
        return cropped_array
    
    # def load_original_data(self, people_no, tgt_index, frame, src_index):
    #     images = {}
    #     depths = {}
    #     masks = {}
    #     ptcs = {}

    #     images['srcs'] = []
    #     images['tgt'] = []
    #     depths['srcs'] = []
    #     depths['tgt'] = []
    #     masks['srcs'] = []
    #     masks['tgt'] = []
    #     ptcs['srcs'] = []

    #     for i in range(2):
    #         src_image_path = self.all_people[people_no][0][src_index[i]][frame]
    #         src_depth_path = self.all_people[people_no][1][src_index[i]][frame]
    #         mask_path = self.all_mask[people_no][src_index[i]][frame]
    #         mask = Image.open(mask_path)
    #         mask = np.array(mask)[:, :, np.newaxis]
    #         mask_crop = self.center_crop_numpy(mask, target_size=self.oral_height)
    #         src_image = Image.open(src_image_path)
    #         src_image = np.array(src_image)[:, :, 0:3]
    #         src_image_crop = self.center_crop_numpy(src_image, target_size=self.oral_height)
    #         src_depth = cv2.imread(str(src_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    #         src_depth_crop = self.center_crop_numpy(src_depth, target_size=self.oral_height)
    #         ptc = self.depth2pts(torch.FloatTensor(src_depth_crop), torch.FloatTensor(self.poses[src_index[0]]), torch.FloatTensor(self.Ks[src_index[0]]))

    #         images['srcs'].append(src_image_crop)
    #         depths['srcs'].append(src_depth_crop)
    #         masks['srcs'].append(mask_crop)
    #         ptcs['srcs'].append(ptc)
        
    #     stereo_data = self.get_rectified_stereo_data(
    #         main_view_data=(images['srcs'][0], masks['srcs'][0], self.Ks[src_index[0]], self.poses[src_index[0]], ptcs['srcs'][0]),
    #         ref_view_data=(images['srcs'][1], masks['srcs'][1], self.Ks[src_index[1]], self.poses[src_index[1]], ptcs['srcs'][1])
    #     )

    #     return 0

        
        
            
    def load_data(self, people_no, tgt_index, frame, src_index):
        images = {}
        depths = {}
        images['srcs'] = []
        images['tgt'] = []
        images['HR_srcs'] = []
        images['original_srcs'] = []
        images['HR_tgt'] = []

        images['original_srcs_masks'] = []
        images['srcs_masks'] = []
        images['HR_srcs_masks'] = []
        images['tgt_mask'] = []
        images['HR_tgt_mask'] = []

        images['original_srcs_vailds'] = []
        images['srcs_vailds'] = []
        images['HR_srcs_vailds'] = []
        images['tgt_vaild'] = []
        images['HR_tgt_vaild'] = []
        

        depths['srcs'] = []
        depths['tgt'] = []
        depths['srcs_est'] = []
        src_image_paths = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        try:
            for i in range(2):
                mask_path = self.all_mask[people_no][src_index[i]][frame]
                mask = Image.open(mask_path)
                mask = np.array(mask) / 255

                valid = (mask.copy() / 255.0).astype(np.float32)
                valid = cv2.erode(valid, kernel, 1)
                valid[valid >= 0.66] = 1.0
                valid[valid < 0.66] = 0.0
                images['original_srcs_vailds'].append(valid[np.newaxis, :, :])

                src_image_path = self.all_people[people_no][0][src_index[i]][frame]
                src_image_paths.append(src_image_path)
                src_depth_path = self.all_people[people_no][1][src_index[i]][frame]
                oral_image = Image.open(src_image_path) * mask[:, :, np.newaxis]
                oral_image = Image.fromarray(oral_image.astype(np.uint8))
                images['original_srcs'].append(np.array(oral_image)[:, :, 0:3].transpose(2, 0, 1))
                images['original_srcs_masks'].append(mask[np.newaxis, :, :])

                image = self.image_exp_transform(oral_image)
                mask_exp = self.image_exp_transform(Image.fromarray((mask * 255).astype(np.uint8)))
                vaild = self.image_exp_transform(Image.fromarray((valid * 255).astype(np.uint8)))
                images['srcs'].append(image)
                images['srcs_masks'].append(mask_exp)
                images['srcs_vailds'].append(vaild)

                exp_HR_srcs = self.image_HR_transform(oral_image)
                exp_HR_srcs_mask = self.image_HR_transform(Image.fromarray((mask * 255).astype(np.uint8)))
                exp_HR_srcs_vailds = self.image_HR_transform(Image.fromarray((valid * 255).astype(np.uint8)))
                images['HR_srcs'].append(exp_HR_srcs)
                images['HR_srcs_masks'].append(exp_HR_srcs_mask)
                images['HR_srcs_vailds'].append(exp_HR_srcs_vailds)

                depth = cv2.imread(str(src_depth_path), cv2.IMREAD_UNCHANGED)
                # depth = depth * mask
                srcs_depth = self.depth_transform(depth)
                depths['srcs'].append(srcs_depth / self.scale)

                depth_est = np.load(src_image_path.with_name(f'{src_image_path.stem}_ed.npy')).astype(np.float32)[0, :, :]
                depth_est = self.depth_transform(depth_est)
                depths['srcs_est'].append(depth_est)
                
            tgt_image_path = self.all_people[people_no][0][tgt_index[0]][frame]
            tgt_depth_path = self.all_people[people_no][1][tgt_index[0]][frame]
        except IndexError:
            print('DEBUG INFO: --------------------------------------------------------')
            print(people_no, frame)
            print(self.all_people[people_no][0][src_index[0]][frame])
            print(self.all_people[people_no][0][src_index[1]][frame])
            print('--------------------------------------------------------------------')
            exit(0)
        
        mask_path = self.all_mask[people_no][tgt_index[0]][frame]
        mask = Image.open(mask_path)
        mask = np.array(mask) / 255

        valid = (mask.copy() / 255.0).astype(np.float32)
        valid = cv2.erode(valid, kernel, 1)
        valid[valid >= 0.66] = 1.0
        valid[valid < 0.66] = 0.0

        oral_image = Image.open(tgt_image_path) * mask[:, :, np.newaxis]
        oral_image = Image.fromarray(oral_image.astype(np.uint8))
        tgt_image = self.image_exp_transform(oral_image)
        tgt_image_mask = self.image_exp_transform(Image.fromarray((mask * 255).astype(np.uint8)))
        tgt_image_vaild = self.image_exp_transform(Image.fromarray((valid * 255).astype(np.uint8)))
        images['tgt'].append(tgt_image)
        images['tgt_mask'].append(tgt_image_mask)
        images['tgt_vaild'].append(tgt_image_vaild)


        HR_tgt_image = self.image_HR_transform(oral_image)
        HR_tgt_image_mask = self.image_HR_transform(Image.fromarray((mask * 255).astype(np.uint8)))
        HR_tgt_image_vaild = self.image_HR_transform(Image.fromarray((valid * 255).astype(np.uint8)))
        images['HR_tgt'].append(HR_tgt_image)
        images['HR_tgt_mask'].append(HR_tgt_image_mask)
        images['HR_tgt_vaild'].append(HR_tgt_image_vaild)

        tgt_depth = cv2.imread(str(tgt_depth_path), cv2.IMREAD_UNCHANGED)
        # tgt_depth = tgt_depth * mask
        tgt_depth = self.depth_transform(tgt_depth)
        depths['tgt'].append(tgt_depth / self.scale)
        return images, depths, src_image_paths
    
    def __getitem__(self, index):
        people_no = index // self.single_people_length
        if self.tgt is None:
            if self.srcs == 4:
                tgt_index = [random.choice([2, 3])]
                src_index = [0, 1]
            elif self.srcs == 3:
                tgt_index = [random.choice([0, 1, 2])]
                src_index = [0, 1]
            else:
                raise Exception('no such data, check your srcs in data_loader_real.py')
        else:
            if self.srcs == 4:
                tgt_index = [self.tgt]
                src_index = [0, 1]
            elif self.srcs == 3:
                tgt_index = [self.tgt]
                src_index = [0, 1]
            else:
                raise Exception('no such data, check your srcs in data_loader_real.py')
        frame = index % self.single_people_length
        Ks, poses, HR_Ks = self.Ks, self.poses, self.HR_Ks
        K = {}
        K['srcs'] = [Ks[i] for i in src_index]
        K['HR_srcs'] = [HR_Ks[i] for i in src_index]
        K['tgt'] = [Ks[tgt_index[i]] for i in range(len(tgt_index))]
        K['HR_tgt'] = [HR_Ks[tgt_index[i]] for i in range(len(tgt_index))]
        pose = {}
        pose['srcs'] = [poses[i] for i in src_index]
        pose['tgt'] = [poses[tgt_index[i]] for i in range(len(tgt_index))]

        images = {}
        depths = {}

        src_image_paths_frames = []
        current_image, current_depth, src_image_paths = self.load_data(people_no=people_no, tgt_index=tgt_index, frame=frame, src_index=src_index)
        # 当前样本的图像和深度详细
        # debug = self.load_original_data(people_no=people_no, tgt_index=tgt_index, frame=frame, src_index=src_index)
        src_image_paths_frames.append(src_image_paths)
        images['srcs'] = current_image['srcs']
        images['tgt'] = current_image['tgt']
        images['HR_srcs'] = current_image['HR_srcs']
        images['HR_tgt'] = current_image['HR_tgt']

        depths['srcs'] = current_depth['srcs']
        depths['srcs_est'] = current_depth['srcs_est']
        depths['tgt'] = current_depth['tgt']
        # 数据增强
        if self.DAtransform is not None:
            images['srcs_DA'] = []
            DA_src0 = self.DAtransform({'image': images['HR_srcs'][0].transpose(1, 2, 0) / 255.0})['image']
            DA_src1 = self.DAtransform({'image': images['HR_srcs'][1].transpose(1, 2, 0) / 255.0})['image']
            images['srcs_DA'] = [DA_src0, DA_src1]
        
        height = self.img_height
        width = self.img_width
        height_width = np.array([height, width])
        K['HW'] = height_width

        return K, pose, images, depths, tgt_index, src_index
        # K, pose, images, depths
    def __len__(self):
        length = int(len(self.all_people)) * self.single_people_length
        return length
    
    def get_rectified_stereo_data(self, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0, pts0 = main_view_data
        img1, mask1, intr1, extr1, pts1 = ref_view_data

        H, W = self.oral_height, self.oral_height
        r0, t0 = extr0[:3, :3], extr0[:3, 3:]
        r1, t1 = extr1[:3, :3], extr1[:3, 3:]
        inv_r0 = r0.T
        inv_t0 = - r0.T @ t0
        E0 = np.eye(4)
        E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
        E1 = np.eye(4)
        E1[:3, :3], E1[:3, 3:] = r1, t1
        E = E1 @ E0
        R, T = E[:3, :3], E[:3, 3]
        dist0, dist1 = np.zeros(4), np.zeros(4)

        intr0 = intr0[:3, :3]
        intr1 = intr1[:3, :3]
        extr0 = extr0[:3, ...]
        extr1 = extr1[:3, ...]

        R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)

        new_extr0 = R0 @ extr0
        new_intr0 = P0[:3, :3]
        new_extr1 = R1 @ extr1
        new_intr1 = P1[:3, :3]
        Tf_x = np.array(P1[0, 3])

        camera = {
            'intr0': new_intr0,
            'intr1': new_intr1,
            'extr0': new_extr0,
            'extr1': new_extr1,
            'Tf_x': Tf_x
        }

        rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
        new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)
        new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        rectify0 = new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y
        rectify1 = new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y

        stereo_data = {
            'img0': new_img0,
            'mask0': new_mask0,
            'img1': new_img1,
            'mask1': new_mask1,
            'camera': camera
        }

        if pts0 is not None:
            flow0, flow1 = self.stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x)

            kernel = np.ones((3, 3), dtype=np.uint8)
            flow_eroded, valid_eroded = [], []
            for (flow, new_mask) in [(flow0, new_mask0), (flow1, new_mask1)]:
                valid = (new_mask.copy()[:, :] / 255.0).astype(np.float32)
                valid = cv2.erode(valid, kernel, 1)
                valid[valid >= 0.66] = 1.0
                valid[valid < 0.66] = 0.0
                flow *= valid
                valid *= 255.0
                flow_eroded.append(flow)
                valid_eroded.append(valid)

            stereo_data.update({
                'flow0': flow_eroded[0],
                'valid0': valid_eroded[0].astype(np.uint8),
                'flow1': flow_eroded[1],
                'valid1': valid_eroded[1].astype(np.uint8)
            })

        return stereo_data

    def stereo_pts2flow(self, pts0, pts1, rectify0, rectify1, Tf_x):
        new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
        new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
        new_depth0 = self.pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
        new_depth1 = self.pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
        new_depth0 = new_depth0.detach().numpy()
        new_depth1 = new_depth1.detach().numpy()
        new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

        offset0 = new_intr1[0, 2] - new_intr0[0, 2]
        disparity0 = -new_depth0 * Tf_x
        flow0 = offset0 - disparity0

        offset1 = new_intr0[0, 2] - new_intr1[0, 2]
        disparity1 = -new_depth1 * (-Tf_x)
        flow1 = offset1 - disparity1

        flow0[new_depth0 < 0.05] = 0
        flow1[new_depth1 < 0.05] = 0

        return flow0, flow1
    
    def pts2depth(self, ptsmap, extrinsic, intrinsic):
        S, S, _ = ptsmap.shape
        pts = ptsmap.view(-1, 3).T
        calib = intrinsic @ extrinsic
        pts = calib[:3, :3] @ pts
        pts = pts + calib[:3, 3:4]
        pts[:2, :] /= (pts[2:, :] + 1e-8)
        depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
        return depth
    
    def depth2pts(self, depth, extrinsic, intrinsic):
        # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
        rot = extrinsic[:3, :3]
        trans = extrinsic[:3, 3:]
        S, S = depth.shape

        y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                            torch.linspace(0.5, S-0.5, S, device=depth.device))
        pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

        pts_2d[..., 2] = 1.0 / (depth + 1e-8)
        pts_2d[..., 0] -= intrinsic[0, 2]
        pts_2d[..., 1] -= intrinsic[1, 2]
        pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
        pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

        pts_2d[..., 0] /= intrinsic[0, 0]
        pts_2d[..., 1] /= intrinsic[1, 1]
        pts_2d = pts_2d.reshape(-1, 3).T
        pts = rot.T @ pts_2d - rot.T @ trans
        return pts.T.view(S, S, 3)
