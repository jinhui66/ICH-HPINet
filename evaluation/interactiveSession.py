import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
import SimpleITK as sitk
from skimage import io, img_as_ubyte
from PIL import Image
from evaluation.metrics import dice
from datetime import datetime
from davisinteractive.utils.scribbles import scribbles2mask
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
import SimpleITK as sitk
class KiTSInteractiveSession:
    """ Class which allows to interface with the evaluation, modified from davisinteractive

    # Arguments
        data_dir: String. Path to the processed KiTS dataset root path.
        ori_data_dir: String. Path to the original KiTS dataset root path, 
            necessary for evaluation on the original size.
        val_set: String. Path to the txt file which contains the index of validation data.
        shuffle: Boolean. Shuffle the samples when evaluating.
        max_time: Integer. Maximum time to evaluate a sample (a sequence with
            a certain set of initial scribbles). This time should be set per
            object as it adapts to the number of objects internally. If
            `max_nb_interactions` is also specified, this time is defined
            as the time available to perform all the interactions.
        max_nb_interactions: Integer. Maximum number of interactions to
            evaluate per sample.
        metric_to_optimize: String. Metric targeting to optimize. Possible values:
            Dice.
        save_report_dir: String. Path to the directory where the report will
            be stored during the evaluation. By default is the current working
            directory.
        save_img_dir: String. Path to the directory where the predicted images will
            be stored during the evaluation. By default is the current working
            directory.
        target_object: String. Target object name, possible values: organ, tumor.
        save_results: Bool. Indicate whether to save the predicted images.
        thresh: Float, range 0-1. Threshold for binarize the masks.
    """
    def __init__(self,
                 data_dir=None,
                 ori_data_dir="/mnts2d/med_data1/lyshi/kits19/data",
                 val_set='/home/shiluyue/Documents/InteractiveCTSeg/Mine/data/test.txt',
                 shuffle=False,
                 max_time=None,
                 max_nb_interactions=5,
                 metric_to_optimize='Dice',
                 save_img_dir=None,
                 save_report_dir=None,
                 target_object="organ",
                 save_results=True,
                 thresh = 0.5
                 ):
        self.data_dir = data_dir
        self.ori_data_dir = ori_data_dir
        self.target_object = target_object
        self.save_results = save_results
        self.thresh = thresh
        if target_object in ["organ","tumor"]:
            self.nb_object = 1

        self.val_set = val_set
        self.shuffle = shuffle
        self.max_time = min(max_time,10 * 60) if max_time is not None else max_time
        self.max_nb_interactions = min(max_nb_interactions,16) if max_nb_interactions is not None else max_nb_interactions
        self.metric_to_optimize = metric_to_optimize

        self.running_model = False
        self.running = True

        self.samples = None
        self.sample_idx = None
        self.interaction_nb = None
        self.sample_start_time = None
        self.sample_scribbles = None
        self.sample_last_scribble = None
        self.interaction_start_time = None
        self.next_scribble_frame_candidates = None

        self.eval_score_slice = []
        self.eval_score_interaction = []
        self.eval_score_sequence = {}
        self.time_interaction = []
        self.time_sequence = {}
        self.eval_original_size_score = {}

        self.save_img_dir = save_img_dir or os.getcwd()
        self.report_save_dir = save_report_dir or os.getcwd()
        if not os.path.exists(self.report_save_dir):
            os.makedirs(self.report_save_dir)

        self.report_name = 'result_%s' % datetime.now().strftime(
            '%Y%m%d_%H%M%S')

        self.global_summary = {}

        '''
        kernel_size: Float. Fraction of the square root of the area used to compute the dilation and erosion before computing the skeleton of the error masks.
        max_kernel_radius: Float. Maximum kernel radius when applying dilation and erosion. Default 16 pixels.
        min_nb_nodes: Integer. Number of nodes necessary to keep a connected graph and convert it into a scribble.
        nb_points: Integer. Number of points to sample the bezier curve when converting the final paths into curves.
        '''
        nb_nodes = 3 if target_object=="tumor" else 4
        self.scribble_robot = InteractiveScribblesRobot(
            kernel_size=0.15,
            max_kernel_radius=16,
            min_nb_nodes=nb_nodes,
            nb_points=1000
        )

        self.pred_mask = None

    def __enter__(self):
        with open(self.val_set) as f:
            id_list = f.readlines()
        self.val_data_index_list = []
        for idx in id_list:
            self.val_data_index_list.append(idx[:-1])
        if self.shuffle:
            self.val_data_index_list = np.random.shuffle(self.val_data_index_list)

        for seq_name in self.val_data_index_list:
            save_dir = os.path.join(self.report_save_dir, seq_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        self.sample_idx = -1
        self.interaction_nb = -1

        self.interaction_slice = []

        return self
 
    def __exit__(self, type, value, trace):
        pass
 
    def next(self):
        """ Iterate to the next iteration/sample of the evaluation process.

        This function moves the iteration to the next iteration or to the next
        sample in case the maximum number of iterations or maximum time have
        been hit.
        This function can be used as control flow on user's code to know until
        which iteration the evuation is being performed.

        # Returns
            bool: Indicates whether the evaluation is still taking place.
        """

        # Here start counter for this interaction, and keep track to move to
        # the next sequence and so on

        c_time = time.time()

        sample_change = self.sample_idx < 0
        if self.max_nb_interactions:
            change_because_interaction = self.interaction_nb >= self.max_nb_interactions
            sample_change |= change_because_interaction
            if change_because_interaction:
                print('Maximum number of interaction have been reached.')
        if self.max_time and self.sample_start_time:
            max_time = self.max_time
            change_because_timing = (c_time - self.sample_start_time) > max_time
            sample_change |= change_because_timing
            if change_because_timing:
                print('Maximum time per sample has been reached.')

        if sample_change:
            # print("Sample Change!")
            self.sample_idx += 1
            self.sample_idx = max(self.sample_idx, 0)
            self.interaction_nb = 0
            if self.sample_idx >= 1: sample_time = c_time - self.sample_start_time
            self.sample_start_time = time.time()
            self.sample_scribbles_list = []

            # self.next_scribble_frame_candidates = [150]

            if self.sample_idx >= 1:
                sequence_name = self.val_data_index_list[self.sample_idx-1]
                sample_report = self.get_sample_report()
                self.interaction_slice = []
                sample_report_filename = os.path.join(self.report_save_dir, sequence_name, '{}_sample_{}.csv'.format(sequence_name,self.report_name))
                sample_report.to_csv(sample_report_filename)
                self.eval_score_sequence[sequence_name] = self.eval_score_interaction
                self.eval_score_interaction = []
                self.time_sequence[sequence_name] = [self.time_interaction,sample_time]
                self.time_interaction = []
            
            if self.sample_idx < len(self.val_data_index_list):
                sequence_name = self.val_data_index_list[self.sample_idx]
                volumn_path = f"{self.data_dir}/preprocessed_image/{sequence_name}.nii.gz"
                mask_path = F"{self.data_dir}/binary_mask/{sequence_name}.nii.gz"
                self.volume = sitk.GetArrayFromImage(sitk.ReadImage(volumn_path))
                self.gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                # self.volume = np.load(volumn_path)["data"]
                # self.gt_mask = self.volume[1]
                self.gt_mask[self.gt_mask >= 0.5] = 1
                self.gt_mask[self.gt_mask < 0.5] = 0


        end = self.sample_idx >= len(self.val_data_index_list)
        if not end and sample_change:
            self.current_seq = self.val_data_index_list[self.sample_idx]
            print("="*20)
            print('sequence {}'.format(self.current_seq))
            print("="*20)

        # Save report on final version if the evaluation ends
        if end:
            print("Saving final report and images.")
            final_report = self.get_final_report()
            final_report_filename = os.path.join(self.report_save_dir, "final_report.csv")
            final_report.to_csv(final_report_filename)
                
            # np.save("/home/shiluyue/Documents/InteractiveCTSeg/Mine/results/debug/pred_mask.npy", self.pred_mask)

            print("Down!")
                    
        return not end
    
    def get_scribbles(self):
        if self.running_model:
            raise RuntimeError(
                'You can not call get_scribbles twice without submitting the '
                'masks first')

        first_ite = False
        if self.interaction_nb == 0:
            first_ite = True

        self.interaction_start_time = time.time()
        self.running_model = True

        data_index = self.current_seq

        # frame_candidates = range(self.pred_mask.shape[0]) if self.next_scribble_frame_candidates is None else self.next_scribble_frame_candidates
        if self.next_scribble_frame_candidates is not None:
            frame_candidates = self.next_scribble_frame_candidates
        else:
            frame_candidates = list(np.unique(np.where(self.gt_mask!=0)[0]))
            if not first_ite:
                frame_candidates += list(np.unique(np.where(self.pred_mask>0.5)[0]))

        if first_ite:
            self.pred_mask = np.zeros(self.gt_mask.shape)
            l = len(frame_candidates)
            selected_slice = np.random.choice(frame_candidates[int(0.1*l):int(0.9*l)])
        else:
            # worst_dice = [100,-1]
            # for i in frame_candidates:
            #     dice_score = dice(self.gt_mask[i], self.pred_mask[i])
            #     if dice_score < worst_dice[0]:
            #         worst_dice = [dice_score, i]
            # selected_slice = worst_dice[1]
            worst = [0,-1]
            for i in frame_candidates:
                gt = self.gt_mask[i]>0.5
                pred = self.pred_mask[i]>0.5
                error = (gt!=pred)
                error_pix_num = np.sum(error)
                if error_pix_num > worst[0]:
                    worst = [error_pix_num, i]
            selected_slice = worst[1]
        
        self.interaction_slice.append(selected_slice)

        self.pred_mask[self.pred_mask>0.5] = 1
        self.pred_mask[self.pred_mask<=0.5] = 0
        # if first_ite:
        #     scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=self.pred_mask,gt_masks=self.gt_mask,frame=selected_slice,nb_objects=self.nb_object)
        # else:
        #     scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=img_as_ubyte(self.pred_mask),
        #                                                     gt_masks=img_as_ubyte(self.gt_mask),frame=selected_slice,nb_objects=self.nb_object)
        # The inputs of scribble_robot have to be uint8(0-255), but when pred_masks is zero, use img_as_ubyte will cause no result, ???.
        scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=self.pred_mask,gt_masks=self.gt_mask,frame=selected_slice,nb_objects=self.nb_object)

        # if self.next_scribble_frame_candidates is None:
        #     scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=self.pred_mask,gt_masks=self.gt_mask, nb_objects=1)
        # elif len(self.next_scribble_frame_candidates) == 1:
        #     scribble_frame = self.next_scribble_frame_candidates[0]
        #     scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=self.pred_mask,gt_masks=self.gt_mask, nb_objects=1,frame=scribble_frame)
        # else:
        #     worst_dice = [100,-1]
        #     for i in self.next_scribble_frame_candidates:
        #         dice_score = dice(self.gt_mask[i], self.pred_mask[i])
        #         if dice_score < worst_dice[0]:
        #             worst_dice = [dice_score, i]
        #     scribble_dict = self.scribble_robot.interact(sequence=data_index, pred_masks=self.pred_mask,gt_masks=self.gt_mask, nb_objects=1,frame=worst_dice[1])
            
        scribble = scribbles2mask(scribble_dict,output_resolution=self.pred_mask.shape[1:])

        self.sample_scribbles_list.append(scribble)

        if first_ite:
            self.save_scribble = scribble + 1
        else:
            self.save_scribble[scribble==0] = 1
            self.save_scribble[scribble==1] = 2

        print('Giving scribble to the user')

        # np.save("/home/shiluyue/Documents/InteractiveCTSeg/Mine/results/debug/pred_mask_{}.npy".format(self.interaction_nb), self.pred_mask)
        if len(np.unique(scribble))==1:
            with open("/home/shiluyue/Documents/InteractiveCTSeg/Mine/results/debug/no_scibble.txt", 'a') as f:
                f.write("{}-{}-{}\n".format(self.current_seq, selected_slice, np.sum(self.gt_mask[selected_slice])))

        return data_index, scribble, first_ite, self.volume
    
    def submit_masks(self, pred_masks, next_scribble_frame_candidates=None):
        """ Submit the predicted masks.

        # Arguments
            pred_masks: Numpy array with the predicted mask for
                the current sample. The array must be of `dtype=int` and
                of size equal to the 480p resolution of the DAVIS
                dataset.
            next_scribble_frame_candidates: List of Integers. Optional value
                specifying the possible frames from which generate the next
                scribble. If values given, the next scribble will be performed
                in the frame where the evaluation metric scores the least on
                the list of given frames. Invalid frames indexes are ignored.
        """
        if not self.running_model:
            raise RuntimeError('You must have called .get_scribbles before '
                               'submiting the masks')

        time_end = time.time()
        self.interaction_nb += 1
        self.running_model = False

        if self.max_time:
            max_t = self.max_time
            if (time_end - self.sample_start_time) > max_t:
                print(
                    ("This submission has been done after the timeout which "
                     "was {}s. This submission won't be evaluated"
                    ).format(max_t))
                return

        timing = time_end - self.interaction_start_time
        self.interaction_start_time = None
        print(
            'The model took {:.3f} seconds to make a prediction'.format(timing))
        self.time_interaction.append(timing)

        self.pred_mask = pred_masks

        self.next_scribble_frame_candidates = next_scribble_frame_candidates 

        self.evaluate_mask(pred_masks)

        interaction_report = self.get_interaction_report()
        interaction_report_filename = os.path.join(self.report_save_dir, self.current_seq, '{}_round{}_{}.csv'.format(self.current_seq,self.interaction_nb,self.report_name))
        interaction_report.to_csv(interaction_report_filename)

        if self.interaction_nb >= self.max_nb_interactions:
            if self.save_results:
                img_save_dir = os.path.join(self.save_img_dir, self.current_seq)
                list_1 = ["ct","pred_mask","gt_mask","scribble"]
                ct = self.volume[0]
                ct = (ct-np.min(ct)) / (np.max(ct) - np.min(ct))
                list_2 = [ct, self.pred_mask, self.gt_mask, self.save_scribble.astype(float)/2]
                target_range = np.unique(np.where(self.gt_mask!=0)[0])
                min_idx = max(np.min(target_range)-5, 0)
                max_idx = min(np.max(target_range)+5, self.gt_mask.shape[0]-1)
                for i in range(4):
                    path = os.path.join(img_save_dir, list_1[i])
                    if not os.path.exists(path):
                        os.makedirs(path)
                    img = list_2[i]
                    for j in range(min_idx, max_idx+1):
                        io.imsave(os.path.join(path, str(j)+".jpg"), img_as_ubyte(img[j]), check_contrast=False)

    def get_interaction_report(self):
        """
        get the dataframe report of the current interaction, which contains the evaluation results of each slice 
        and the whole volumn.
        """
        slices = ["Slice%03d"%i for i in range(len(self.eval_score_slice))] + ["Volumn"]
        scores = self.eval_score_slice + self.eval_score_interaction[-1:]
        times = [0 for i in range(len(self.eval_score_slice))] + self.time_interaction[-1:]
        interaction = [0 for i in range(len(self.eval_score_slice)+1)]
        interaction[self.interaction_slice[self.interaction_nb-1]] = 1
        data = {"Slice":slices, "Score":scores, "Time":times, "Interaction Slice":interaction}
        df = pd.DataFrame(data, index=None)
        return df
    
    def get_sample_report(self):
        """
        get the dataframe report of the current sequence, which contains the evaluation results and processing times 
        of each interaction.
        """
        rounds = ["Round_%02d"%(i+1) for i in range(len(self.eval_score_interaction))]
        scores = self.eval_score_interaction
        times = np.round(self.time_interaction, 2)
        data = {"Round":rounds, "Score":scores, "Time":times, "Interaction Slice":self.interaction_slice}
        df = pd.DataFrame(data, index=None)

        # evaluating on the original size
        seg_path = os.path.join(self.ori_data_dir, self.current_seq, "segmentation.nii.gz")
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        gt = sitk.GetArrayFromImage(seg).transpose(2,0,1)
        if self.target_object=="organ":
            gt[gt!=1] = 0
        else:
            gt[gt!=2] = 0
        z, h, w = gt.shape
        pred = torch.from_numpy(self.pred_mask)
        pred = pred.unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=[z,h,w], mode="trilinear", align_corners=False)
        pred = pred[0,0].numpy()
        score = dice(gt, pred, thresh=self.thresh)

        self.eval_original_size_score[self.current_seq] = score

        return df

    def get_final_report(self):
        """
        get the final dataframe report, which contains the evaluation results and processing times 
        of each sequence.
        """
        sequences = self.val_data_index_list
        scores = [self.eval_score_sequence[s][-1] for s in sequences]
        times_1 = [np.sum(self.time_sequence[s][0]) for s in sequences]
        times_2 = [self.time_sequence[s][1] for s in sequences]
        scores_ori = [self.eval_original_size_score[s] for s in sequences]

        data = {"Case":sequences, "Score":scores, "Score_Original":scores_ori, "Time_1":np.round(times_1,2), "Time_2":np.round(times_2,2)}
        df = pd.DataFrame(data, index=None)
        return df

    def evaluate_mask(self, pred_mask):
        """
        calculate the evaluation results
        """
        self.eval_score_slice = []
        if self.metric_to_optimize == "Dice":
            sample_dice = dice(self.gt_mask, pred_mask, thresh=self.thresh)
            self.eval_score_interaction.append(sample_dice)
            for i in range(pred_mask.shape[0]):
                slice_dice = dice(self.gt_mask[i], pred_mask[i], thresh=self.thresh)
                self.eval_score_slice.append(slice_dice)
        else:
            print("The evaluation metric does not exist, choose from 'Dice'.")    


# mask_dir="."
# val_set='/home/shiluyue/Documents/InteractiveCTSeg/Mine/data/test_2.txt'
# shuffle=False
# max_time=None
# max_nb_interactions=5
# metric_to_optimize='Dice'
# report_save_dir="/home/shiluyue/Documents/InteractiveCTSeg/Mine/results"

# with KiTSInteractiveSession(mask_dir=mask_dir,val_set=val_set,shuffle=shuffle,max_time=max_time,
#                             max_nb_interactions=max_nb_interactions,metric_to_optimize=metric_to_optimize,
#                             report_save_dir=report_save_dir) as sess:
#     while sess.next():
#         data_index, scribble, first_ite = sess.get_scribbles()
#         pred_masks = np.ones((10,128,128))
#         sess.submit_masks(pred_masks)


