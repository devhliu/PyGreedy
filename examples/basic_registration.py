from pygreedy import GreedyRegistration
from pygreedy.utils import load_image, save_image

def main():
    # Load images
    fixed_data, fixed_affine = load_image("fixed.nii.gz")
    moving_data, moving_affine = load_image("moving.nii.gz")
    
    # Create registration object
    reg = GreedyRegistration(
        iterations=200,
        sigma=2.0,
        learning_rate=0.1,
        metric="ncc",
        use_gpu=True,
        verbose=True
    )
    
    # Perform registration
    result = reg.register(fixed_data, moving_data)
    
    # Save results
    save_image(
        "warped.nii.gz",
        result['warped_image'],
        fixed_affine,
        reference="fixed.nii.gz"
    )
    
    print("Registration completed successfully!")

if __name__ == "__main__":
    main()