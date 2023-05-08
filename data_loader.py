import numpy as np
# import zipfile

class PongDataLoader():
    def __init__(self,zip_name = 'dataset40.zip'):
        self.zip_name = zip_name
        # self.datasetzip = zipfile.ZipFile(zip_name)
        # self.all_names = self.datasetzip.namelist()
        self.numpy_files = np.load(zip_name)
        self.all_names = list(self.numpy_files.keys())

        self.episode_list = []
        for file_name in self.all_names:
            if file_name[-5:]=='state':
                ep = int(file_name[4:7])
                if ep not in self.episode_list:
                    self.episode_list.append(ep)
        self.episode_list = sorted(self.episode_list)

        self.step_list = []
        ep = self.episode_list[0]
        for file_name in self.all_names:
            if file_name[-5:]=='state':
                ep_temp = int(file_name[4:7])
                if ep==ep_temp:
                    step = int(file_name[13:16])
                    self.step_list.append(step)
        self.step_list = sorted(self.step_list)
        # sanity check
        for sanity_ep in self.episode_list:
            sanity_step_list = []
            for file_name in self.all_names:
                if file_name[-5:]=='state':
                    ep_temp = int(file_name[4:7])
                    if sanity_ep==ep_temp:
                        step = int(file_name[13:16])
                        sanity_step_list.append(step)
            sanity_step_list = sorted(sanity_step_list)
            if sanity_step_list != self.step_list:
                raise ReferenceError('The file has episodes with unequal step numbers.\
                                     \n Different step sizes for each episode is not supported yet')
        self.episode_list = np.array(self.episode_list)
        self.step_list = np.array(self.step_list)

        

        # extract shape
        i = 0
        while True:
            file_name = self.all_names[i]
            if file_name[-5:] == 'state':
                my_array = self.numpy_files[file_name]
                self.first_dim,self.second_dim = my_array.shape
                break
            else: 
                i+=1
        

        

    def get_data(self,num_sequences=16,sequence_length=15):
        state_sequences = np.zeros([num_sequences,sequence_length,self.first_dim,self.second_dim])
        action_sequences = np.zeros([num_sequences,sequence_length])
        randomly_chosen_episodes = np.random.choice(self.episode_list,size=num_sequences,replace=True)
        for k,episode in enumerate(randomly_chosen_episodes):
            randomly_chosen_step = np.random.choice(self.step_list[sequence_length:],1)
            for i in np.arange(sequence_length-1,0,-1):
                state_file_name = 'ep%05dstep%05d_state'%(episode,randomly_chosen_step-i)
                state_sequences[k,i,:,:] = self.numpy_files[state_file_name]
                action_file_name = 'ep%05dstep%05d_action'%(episode,randomly_chosen_step-i)
                action_sequences[k,i] = self.numpy_files[action_file_name]
        return state_sequences,action_sequences