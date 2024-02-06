import cv2
from image_processor import process_image
from text_remover import remove_text

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    # Process the image
    processed_image = process_image(image)

    # Remove text from image
    text_removed_image = remove_text(processed_image)

    # Display the result
    cv2.imshow('Text Removed Image', text_removed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test\test_img.png")
