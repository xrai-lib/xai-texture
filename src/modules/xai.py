from utils import clear_screen
from xai.GLCM import analyse_GLCM
from xai.GLCM_Unet import analyze_GLCM_Unet
from xai.GLCM_Unet import analyze_GLCM_Hrnet
from xai.LTEM_Unet import LTEM_analysis_unet
from xai.LTEM_hrnet import LTEM_analysis_hrnet

from xai.LTEM_Cosine_Similarity import cosine_similarity_analysis

def prompt_analysis():
    print("1. GLCM feature Map Analysis")
    print("2. LTEM feature Map Cosine Similarity Analysis")
    print("3. Exit")

    choice = None
    while True:
            try:
                choice = int(input("Select Function (1-3): "))
                if 1 <= choice <= 3:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 3 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice

def prompt_model():
    clear_screen()

    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")
    print("4. HR-Net")
    print("5. FPN-Net")
    print("6. Link-Net")

    choice_model = None
    while True:
            try:
                choice_model = int(input("Select Model (1-6): "))
                if 1 <= choice_model <= 6:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice_model

def prompt_dataset():

    print("1. CBIS_DDSM")
    print("2. CBIS_DDSM_CLAHE")
    print("3. HAM10000")
    print("4. HAM10000_CLAHE")
    print("5. POLYP")
    print("6. POLYP_CLAHE")
    

    choice_dataset = None
    while True:
            try:
                choice_dataset = int(input("Select Dataset (1-6): "))
                if 1 <= choice_dataset <= 6:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 6 available datasets.")
            except ValueError:
                print("That's not an integer. Please try again.")

    return choice_dataset

def xai_module():
    clear_screen() #clears the terminal screen
    print("Welcome to the Texture Analysis Module")
    
    choice_analysis = prompt_analysis()
    choice_model = prompt_model()
    choice_dataset = prompt_dataset()

    #perform user requested tasks
    while choice_analysis != 3:
        
        if choice_analysis == 1:
            if choice_model == 3:
                analyze_GLCM_Unet(choice_dataset)
            elif choice_model == 4:
                analyze_GLCM_Hrnet(choice_dataset)
            else:
                analyse_GLCM(choice_model, choice_dataset)

        elif choice_analysis == 2:
            if choice_model == 3:
                LTEM_analysis_unet(choice_dataset)
            elif choice_model == 4:
                LTEM_analysis_hrnet(choice_dataset)
            else: 
                cosine_similarity_analysis(choice_model, choice_dataset)
            
        choice_analysis = prompt_analysis()

    return