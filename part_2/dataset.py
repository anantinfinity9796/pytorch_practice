import collections
import copy
import numpy as np
import functools
import os
import glob
from collections import namedtuple
import csv
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from multiprocessing import Manager



IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x','y','z'])
candidate_info_tuple = namedtuple('candidate_info_tuple', ['isNodule_bool', 'diameter_mm', 'series_uid','center_xyz'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    # Swipes the order of the coordinates from I,R,C to C,R,I while we convert to a numpy array of the voxels coordinates to x,y,z coordinates
    cri_a = np.array(coord_irc)[::-1]
    # Converts the origin to an array
    origin_a = np.array(origin_xyz)
    # converts the voxel size to an array
    vxSize_a = np.array(vxSize_xyz)

    # Multiplies the indices with the voxel_szie to scale them and matrix multiply with the directions so that it is captured aacordingly in each axis and offset the origin.
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)  # Convert it into a named tuple and return it

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a =  np.array(coord_xyz)

    # Inverse  of the last 3 steps
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a))/vxSize_a

    # rounds off before converting to integers
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0])) # Converts from shape C,R,I to I,R,C and converts to integers


@functools.lru_cache(1)
def GetCandidateInfoList(requireOnDisk_bool = True):
    
    # get a list of all the .mhd files in the various subsets of the data.
    mhd_list = glob.glob("E://data/data-unversioned/subset*/*.mhd")

    # We are splitting the whole filepath into its various pieces, taking only the filename(hence the -1) and removing the .mhd from it(hence the -4). 
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # here we will define a diameter dict to keep track of distance between individual nodules.
    # If they are really close, treat them as a single nodule and save their series_uid along with their diameter, else 0 diameter
    diameter_dict = {}

    with open("E://data/data-unversioned/annotations.csv", 'r') as f:

        for row in list(csv.reader(f))[1:]:  # starting from the first row becuse row 0 is just column names(headers)
            # get the series_uid of the nodule which is the first value of the row
            series_uid = row[0]

            # get the x,y,z coordinates(which are the 2,3,4 values respectively) which mark the center of the nodules.
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])

            # Get the diameter of the nodules which is the 5th value of each row
            annotation_diameter_mm = float(row[4])

            # Set the default of the key i.e series_uid as an empty list and this also returns the value of that key.
            # Then append a tuple of the x,y,z coordinates of the center of the nodule and the diameter of the annotated nodule
            diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))

    # Now  we will build a full list of candidates nodules using the information in candidates.csv file
    candidate_info_list = []
    with open("E://data/data-unversioned/candidates.csv", 'r') as f:
        for row in list(csv.reader(f))[1:]:

            # Get the series_uid
            series_uid = row[0]

            # If the series_uid is not present on disk and the requireOnDisk attribute is set to True then skip the file
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            
            # Put if the candidate is a nodule or not into the isNodule_bool parameter. This is the 5th value of the row
            isNodule_bool = bool(int(row[4]))

            # Get the x,y,z coordinates of the center of the candidate
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            # set the candidate diameter to 0.0 
            candidate_diameter_mm = 0.0

            # loop over the annotation_tuple dictionary and get the annotation tuple of the matching series_uid of the candidate
            for annotation_tuple in diameter_dict.get(series_uid, []):
                # get the x,y,z center values of the dictionary and the annotation_diameter of the annotated tuple
                # print(diameter_dict.get(series_uid, []), annotation_tuple)
                # print(type(annotation_tuple), len(annotation_tuple))
                annotation_center_xyz, annotation_diameter_mm = annotation_tuple

                # Now this loops over the x,y,z coodinates and finds the absolute distance between the centers of annotated and candidate nodules.
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])

                    # if delta is > annotation_diameter/4
                    # annotation_diameter/4 --> Divides it by 2 to get diameter, again divides it by 2 to get the radius and then compares.
                    # This is done to make sure that the centers are not too far apart relative to the size of the nodule.
                    # This is a type of a bounding box check and not a distance check.
                    if delta_mm > annotation_diameter_mm /4:
                        # If the candidates and the annotations are not close then break and add them to the candidate list as seperate nodules
                        break
                else:
                    # If they are very close then we should see them as the same nodule and then the diameter would be the annotation diameter
                    candidate_diameter_mm = annotation_diameter_mm
                    break

            # Covert all the in information into a tuple and append it to the candidate list
            candidate_info_list.append(candidate_info_tuple(isNodule_bool,
                                                            candidate_diameter_mm,
                                                            series_uid,
                                                            candidate_center_xyz))

    # Sort the list in ascending/descending order to make the sampling representative of the dataset.
    candidate_info_list.sort(reverse=True)
    return candidate_info_list

class Ct():
    def __init__(self, series_uid):
        mhd_path = glob.glob(f"E://data/data-unversioned/subset*/{series_uid}.mhd")[0]
        
        # The readImage automatically consumes the .raw file in addition to the .mhd file passed in
        ct_mhd = sitk.ReadImage(mhd_path)

        # Recreates an np.array since we want to convert the value type to np.float32
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32)
        ct_a.clip(-1000,1000,ct_a)

        # All of the values we have built above are now assigned to self to make them the attributes of the object
        self.series_uid = series_uid
        self.hu_a = ct_a
        
        # Get the origin values from the .mhd file which would be used to convert from x,y,z coordinates to the i,r,c coordinates
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())

        # Get the size of the voxel which needs to be multiplied to every axis to get the correct voxel length from xyz to irc of each axis
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())

        # converts the directions to an array, and reshapes the nine-element array to 3,3
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)
        # These are the inputs we need to pass into our xyz2irc conversions In addition to the individual point to convert.
        # with these attributes, our CT object implementation now has all the data needed to convert a candidate center from patient coordinates to array coordinates.
    

    def getRawCandidate(self, center_xyz, width_irc):
        """ This function crops the relevant voxels(bounding box of the nodule) from the full CT scan array.
            It does so by converting from x,y,z to i,r,c of the center of the nodule and then using the width to calculate the
            start and end indexes of the crop box.  The width_irc is a fixed 3D width(which resembles our input shape) with which
            the CT scan will need to be cropped so that the center of the cropped array and the nodule are aligned."""
        center_irc = xyz2irc(
            coord_xyz = center_xyz, # center corrdinates of the ct_array
            origin_xyz = self.origin_xyz, # origin information of the ct_array
            vxSize_xyz = self.vxSize_xyz, # voxel_size of the ct array
            direction_a = self.direction_a # direction matrix of the ct_array
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            # Find the start and the end indexes for every axis
            start_ndx = int(round(center_val-width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            # Append the start and the end indexes for every axis onto a list
            slice_list.append(slice(start_ndx, end_ndx))
        
        # Now crop the CT scan with the start and end indexes list to get the cropped 3D CT array which has the nodule clearly centered
        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc   # Return the CT chunk and the I,R,C corrdinates of the center.
        



@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@functools.lru_cache(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self, val_stride = 0, is_val_set_bool = None, series_uid = None):
        candidateInfo_list = copy.copy(GetCandidateInfoList())
        manager = Manager()
        self.candidateInfo_list = manager.list(candidateInfo_list)


        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list


    def __len__(self):
        return len(self.candidateInfo_list)


    
    def __getitem__(self, ndx):
        candidate_info_tup = self.candidateInfo_list[ndx]
        # print(candidate_info_tup)
        # assert isinstance(candidate_info_tup, collections.namedtuple)
        width_irc = (32,48,48)

        candidate_a, center_irc = getCtRawCandidate(
            candidate_info_tup.series_uid, 
            candidate_info_tup.center_xyz,
            width_irc
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidate_info_tup.isNodule_bool,
            candidate_info_tup.isNodule_bool
        ],
        dtype = torch.long)

        return (candidate_t, 
                pos_t,
                candidate_info_tup.series_uid,
                torch.tensor(center_irc))



