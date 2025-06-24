import numpy as np
from typing import Union, List, Tuple, Optional

class BrailleUtils:
    """Utility class for converting images to braille representations and displaying training samples."""
    
    def __init__(self):
        """Initialize BrailleUtils with complete braille pattern set."""
        self.braille_patterns = self._generate_braille_patterns()
    
    @staticmethod
    def _generate_braille_patterns():
        """Generate all possible braille patterns"""
        patterns = {}
        for i in range(256):
            # Convert to braille unicode
            braille_char = chr(0x2800 + i)
            patterns[i] = braille_char
        return patterns
    
    @staticmethod
    def _white_text(text):
        """Return text in white color"""
        return f"\033[97m{text}\033[0m"
    
    @staticmethod
    def _green_text(text):
        """Return text in green color"""
        return f"\033[92m{text}\033[0m"
    
    @staticmethod
    def _red_text(text):
        """Return text in red color"""
        return f"\033[91m{text}\033[0m"
    
    @staticmethod
    def extract_2x2_digits_from_stacked_image(stacked_image: np.ndarray) -> np.ndarray:
        """
        Extract 4 individual digit images from a 56x56 stacked image.
        
        Args:
            stacked_image: (56, 56) numpy array containing 2x2 grid of digits
            
        Returns:
            (4, 28, 28) numpy array with individual digit images
            Order: [top_left, top_right, bottom_left, bottom_right]
        """
        if stacked_image.shape != (56, 56):
            raise ValueError(f"Expected (56, 56) stacked image, got {stacked_image.shape}")
        
        # Split into 4 28x28 digit images
        top_left = stacked_image[:28, :28]
        top_right = stacked_image[:28, 28:]
        bottom_left = stacked_image[28:, :28]
        bottom_right = stacked_image[28:, 28:]
        
        return np.array([top_left, top_right, bottom_left, bottom_right])
    
    def display_stacked_image_sample(self,
                                   stacked_image: np.ndarray,
                                   expected_digits: List[int],
                                   predicted_digits: Optional[List[int]] = None,
                                   epoch: Optional[int] = None,
                                   batch_idx: Optional[int] = None,
                                   sample_idx: Optional[int] = None,
                                   threshold: float = 0.5) -> None:
        """
        Display a 56x56 stacked image sample with 2x2 digit extraction and labeling.
        
        Args:
            stacked_image: (56, 56) numpy array containing 2x2 grid of digits
            expected_digits: List of 4 expected digit labels [top-left, top-right, bottom-left, bottom-right]
            predicted_digits: Optional list of 4 predicted digit labels
            epoch: Optional epoch number for display
            batch_idx: Optional batch index for display
            sample_idx: Optional sample index for display
            threshold: Threshold for binary conversion
        """
        # Extract individual digit images using 2x2 layout logic (silently)
        digit_images = self.extract_2x2_digits_from_stacked_image(stacked_image)
        
        # Use existing display method
        self.display_training_samples(
            digit_images,        # (4, 28, 28) - individual digit images
            expected_digits,     # [4] - expected labels
            predicted_digits,    # [4] - predicted labels
            epoch=epoch,
            batch_idx=batch_idx,
            sample_idx=sample_idx,
            threshold=threshold
        )
    
    def display_batch_samples(self,
                            stacked_images: np.ndarray,
                            expected_labels: np.ndarray,
                            predicted_labels: np.ndarray,
                            epoch: Optional[int] = None,
                            max_samples: int = 10,
                            threshold: float = 0.5) -> None:
        """
        Display multiple stacked image samples from a batch with 2x2 grid extraction.
        
        Args:
            stacked_images: (B, 56, 56) numpy array of stacked images
            expected_labels: (B, 4) numpy array of expected digit labels
            predicted_labels: (B, 4) numpy array of predicted digit labels
            epoch: Optional epoch number for display
            max_samples: Maximum number of samples to display
            threshold: Threshold for binary conversion
        """
        batch_size = min(stacked_images.shape[0], max_samples)
        
        # Display legend once at the beginning
        print(f"\nLegend: {self._white_text('Expected')} | {self._green_text('Predicted Correct')} | {self._red_text('Predicted Wrong')}")
        print("=" * 70)
        
        for sample_idx in range(batch_size):
            # Extract 2x2 layout and display (silently)
            self.display_stacked_image_sample(
                stacked_images[sample_idx],           # Single 56x56 stacked image
                expected_labels[sample_idx].tolist(), # 4 expected digits [TL, TR, BL, BR]
                predicted_labels[sample_idx].tolist(), # 4 predicted digits [TL, TR, BL, BR]
                epoch=epoch,
                sample_idx=sample_idx + 1,
                threshold=threshold
            )
    
    def image_to_braille(self, image: np.ndarray, threshold: float = 0.5) -> str:
        """
        Convert a 28x28 image to braille representation.
        
        Args:
            image: 28x28 numpy array (grayscale values 0-1 or 0-255)
            threshold: threshold for converting to binary
        
        Returns:
            String representation using braille characters
        """
        # Normalize image to 0-1 range
        if image.max() > 1:
            image = image / 255.0
        
        # Convert to binary
        binary_image = (image > threshold).astype(int)
        
        # Pad image to be divisible by 4x2 blocks (braille cells)
        h, w = binary_image.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (2 - w % 2) % 2
        
        if pad_h > 0 or pad_w > 0:
            binary_image = np.pad(binary_image, ((0, pad_h), (0, pad_w)), mode='constant')
        
        result_lines = []
        
        # Process in 4x2 blocks (braille cells)
        for row in range(0, binary_image.shape[0], 4):
            line = ""
            for col in range(0, binary_image.shape[1], 2):
                # Extract 4x2 block
                block = binary_image[row:row+4, col:col+2]
                
                # Map to braille pattern
                pattern = 0
                if block.shape[0] > 0 and block.shape[1] > 0:
                    if block[0, 0]: pattern |= 0b00000001  # dot 1
                if block.shape[0] > 1 and block.shape[1] > 0:
                    if block[1, 0]: pattern |= 0b00000010  # dot 2
                if block.shape[0] > 2 and block.shape[1] > 0:
                    if block[2, 0]: pattern |= 0b00000100  # dot 3
                if block.shape[0] > 3 and block.shape[1] > 0:
                    if block[3, 0]: pattern |= 0b01000000  # dot 7
                if block.shape[0] > 0 and block.shape[1] > 1:
                    if block[0, 1]: pattern |= 0b00001000  # dot 4
                if block.shape[0] > 1 and block.shape[1] > 1:
                    if block[1, 1]: pattern |= 0b00010000  # dot 5
                if block.shape[0] > 2 and block.shape[1] > 1:
                    if block[2, 1]: pattern |= 0b00100000  # dot 6
                if block.shape[0] > 3 and block.shape[1] > 1:
                    if block[3, 1]: pattern |= 0b10000000  # dot 8
                
                line += self.braille_patterns[pattern]
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def display_2x2_digits_with_labels(self, 
                                     digits: List[np.ndarray],
                                     expected_digits: List[int],
                                     predicted_digits: Optional[List[int]] = None,
                                     digit_size: int = 14, 
                                     threshold: float = 0.5) -> str:
        """
        Display a 2x2 grid of digits with colored labels.
        White = Expected, Green = Predicted Correct, Red = Predicted Wrong
        
        Args:
            digits: List of 4 digit image arrays
            expected_digits: List of 4 expected digit labels
            predicted_digits: Optional list of 4 predicted digit labels
            digit_size: Size of each digit display
            threshold: Threshold for binary conversion
        
        Returns:
            String representation of 2x2 grid with colored labels
        """
        if len(digits) != 4:
            raise ValueError("Must provide exactly 4 digit images")
        if len(expected_digits) != 4:
            raise ValueError("Must provide exactly 4 expected digit labels")
        if predicted_digits is not None and len(predicted_digits) != 4:
            raise ValueError("Must provide exactly 4 predicted digit labels or None")
        
        # Convert digits to braille strings
        digit_brailles = []
        for i, digit in enumerate(digits):
            # Convert image to braille
            if digit.shape[0] != digit_size or digit.shape[1] != digit_size:
                digit = self._resize_image(digit, (digit_size, digit_size))
            braille_str = self.image_to_braille(digit, threshold)
            
            # Add labels to each cell
            braille_lines = braille_str.split('\n')
            
            # Add expected digit (white) at top-left of first line
            expected_label = self._white_text(f"{expected_digits[i]}")
            if braille_lines:
                braille_lines[0] = expected_label + braille_lines[0][1:]  # Replace first character
            
            # Add predicted digit (green if correct, red if wrong) at bottom-right if provided
            if predicted_digits is not None:
                is_correct = predicted_digits[i] == expected_digits[i]
                if is_correct:
                    predicted_label = self._green_text(f"{predicted_digits[i]}")
                else:
                    predicted_label = self._red_text(f"{predicted_digits[i]}")
                
                if braille_lines:
                    last_line = braille_lines[-1]
                    braille_lines[-1] = last_line[:-1] + predicted_label  # Replace last character
            
            digit_brailles.append('\n'.join(braille_lines))
        
        # Combine into 2x2 layout
        top_left, top_right, bottom_left, bottom_right = digit_brailles
        
        # Split each braille representation into lines
        tl_lines = top_left.split('\n')
        tr_lines = top_right.split('\n')
        bl_lines = bottom_left.split('\n')
        br_lines = bottom_right.split('\n')
        
        # Combine top row
        result_lines = []
        max_top_lines = max(len(tl_lines), len(tr_lines))
        
        for i in range(max_top_lines):
            left_part = tl_lines[i] if i < len(tl_lines) else ' ' * 10  # Approximate width
            right_part = tr_lines[i] if i < len(tr_lines) else ' ' * 10
            result_lines.append(left_part + ' | ' + right_part)
        
        # Add separator
        separator_width = len(result_lines[0]) if result_lines else 25
        result_lines.append('-' * separator_width)
        
        # Combine bottom row
        max_bottom_lines = max(len(bl_lines), len(br_lines))
        
        for i in range(max_bottom_lines):
            left_part = bl_lines[i] if i < len(bl_lines) else ' ' * 10
            right_part = br_lines[i] if i < len(br_lines) else ' ' * 10
            result_lines.append(left_part + ' | ' + right_part)
        
        return '\n'.join(result_lines)
    
    def display_2x2_digits(self, digits: List[Union[np.ndarray, int]], 
                          digit_size: int = 14, 
                          threshold: float = 0.5) -> str:
        """
        Display a 2x2 grid of digits using braille patterns.
        
        Args:
            digits: List of 4 elements (top-left, top-right, bottom-left, bottom-right)
                    Each can be a digit image array or integer label
            digit_size: Size of each digit display (default 14x14)
            threshold: Threshold for binary conversion
        
        Returns:
            String representation of 2x2 grid in braille
        """
        if len(digits) != 4:
            raise ValueError("Must provide exactly 4 digits for 2x2 grid")
        
        # Convert digits to braille strings
        digit_brailles = []
        for digit in digits:
            if isinstance(digit, (int, np.integer)):
                # Create a simple text representation for integer labels
                digit_str = str(digit)
                digit_brailles.append(f"Digit: {digit_str}")
            else:
                # Convert image to braille
                if digit.shape[0] != digit_size or digit.shape[1] != digit_size:
                    # Resize if needed (simple nearest neighbor)
                    digit = self._resize_image(digit, (digit_size, digit_size))
                digit_brailles.append(self.image_to_braille(digit, threshold))
        
        # Combine into 2x2 layout
        top_left, top_right, bottom_left, bottom_right = digit_brailles
        
        # Split each braille representation into lines
        tl_lines = top_left.split('\n')
        tr_lines = top_right.split('\n')
        bl_lines = bottom_left.split('\n')
        br_lines = bottom_right.split('\n')
        
        # Combine top row
        result_lines = []
        max_top_lines = max(len(tl_lines), len(tr_lines))
        
        for i in range(max_top_lines):
            left_part = tl_lines[i] if i < len(tl_lines) else ' ' * len(tl_lines[0])
            right_part = tr_lines[i] if i < len(tr_lines) else ' ' * len(tr_lines[0])
            result_lines.append(left_part + ' | ' + right_part)
        
        # Add separator
        separator_width = len(result_lines[0]) if result_lines else 20
        result_lines.append('-' * separator_width)
        
        # Combine bottom row
        max_bottom_lines = max(len(bl_lines), len(br_lines))
        
        for i in range(max_bottom_lines):
            left_part = bl_lines[i] if i < len(bl_lines) else ' ' * len(bl_lines[0])
            right_part = br_lines[i] if i < len(br_lines) else ' ' * len(br_lines[0])
            result_lines.append(left_part + ' | ' + right_part)
        
        return '\n'.join(result_lines)
    
    def display_training_samples(self, 
                               generated_images: np.ndarray,
                               expected_digits: List[int],
                               predicted_digits: Optional[List[int]] = None,
                               epoch: Optional[int] = None,
                               batch_idx: Optional[int] = None,
                               sample_idx: Optional[int] = None,
                               threshold: float = 0.5) -> None:
        """
        Display 2x2 generated sample images with expected and predicted digits for encoder-decoder training.
        
        Args:
            generated_images: Array of shape (4, H, W) containing 4 generated images
            expected_digits: List of 4 expected digit labels
            predicted_digits: Optional list of 4 predicted digit labels
            epoch: Optional epoch number for display
            batch_idx: Optional batch index for display
            sample_idx: Optional sample index for display
            threshold: Threshold for binary conversion
        """
        if generated_images.shape[0] != 4:
            raise ValueError("Must provide exactly 4 generated images")
        if len(expected_digits) != 4:
            raise ValueError("Must provide exactly 4 expected digit labels")
        
        # Create header
        header_parts = []
        if epoch is not None:
            header_parts.append(f"Epoch {epoch}")
        if batch_idx is not None:
            header_parts.append(f"Batch {batch_idx}")
        if sample_idx is not None:
            header_parts.append(f"Sample {sample_idx}")
        if header_parts:
            header = " - ".join(header_parts)
        else:
            header = "Training Sample"
        
        print(f"\n{header}")
        print("=" * len(header))
        
        # Display with colored labels
        braille_display = self.display_2x2_digits_with_labels(
            list(generated_images), 
            expected_digits,
            predicted_digits,
            digit_size=generated_images.shape[1], 
            threshold=threshold
        )
        print(braille_display)
    
    @staticmethod
    def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest neighbor resize"""
        h, w = image.shape
        target_h, target_w = target_size
        
        # Create coordinate arrays
        y_indices = np.round(np.linspace(0, h-1, target_h)).astype(int)
        x_indices = np.round(np.linspace(0, w-1, target_w)).astype(int)
        
        # Use advanced indexing to resize
        resized = image[np.ix_(y_indices, x_indices)]
        return resized
    
    def display_image_braille(self, image: np.ndarray, title: str = None) -> None:
        """
        Display an image as braille in the terminal.
        
        Args:
            image: 2D numpy array representing the image
            title: Optional title to display above the image
        """
        if title:
            print(f"\n{title}")
            print("=" * len(title))
        
        braille_str = self.image_to_braille(image)
        print(braille_str)
        print()
    
    def generate_test_digit(self, digit: int) -> np.ndarray:
        """
        Generate and return a MNIST-style 28x28 digit pattern.
        
        Args:
            digit: Integer digit (0-9) to generate
            
        Returns:
            28x28 numpy array representing the digit
        """
        if not 0 <= digit <= 9:
            raise ValueError("Digit must be between 0 and 9")
        
        # Create a 28x28 canvas
        canvas = np.zeros((28, 28))
        
        # Simple digit patterns (approximating MNIST style)
        if digit == 0:
            # Oval shape
            for i in range(4, 24):
                for j in range(8, 20):
                    if ((i-14)**2/100 + (j-14)**2/36) <= 1 and ((i-14)**2/64 + (j-14)**2/16) >= 1:
                        canvas[i, j] = 1.0
        
        elif digit == 1:
            # Vertical line with top stroke
            canvas[6:26, 13:15] = 1.0
            canvas[6:10, 10:14] = 1.0
            canvas[22:26, 10:18] = 1.0
        
        elif digit == 2:
            # S-curve shape
            canvas[6:10, 8:20] = 1.0
            canvas[10:14, 16:20] = 1.0
            canvas[14:18, 8:16] = 1.0
            canvas[18:22, 8:12] = 1.0
            canvas[22:26, 8:20] = 1.0
        
        elif digit == 3:
            # Two horizontal bars connected on right
            canvas[6:10, 8:20] = 1.0
            canvas[12:16, 8:20] = 1.0
            canvas[22:26, 8:20] = 1.0
            canvas[10:22, 16:20] = 1.0
        
        elif digit == 4:
            # Vertical lines with horizontal connector
            canvas[6:18, 8:12] = 1.0
            canvas[14:18, 8:20] = 1.0
            canvas[16:26, 16:20] = 1.0
        
        elif digit == 5:
            # Reverse S shape
            canvas[6:10, 8:20] = 1.0
            canvas[10:14, 8:12] = 1.0
            canvas[14:18, 8:20] = 1.0
            canvas[18:22, 16:20] = 1.0
            canvas[22:26, 8:20] = 1.0
        
        elif digit == 6:
            # Circle with gap at top right
            canvas[10:26, 8:12] = 1.0
            canvas[14:18, 8:20] = 1.0
            canvas[18:26, 16:20] = 1.0
            canvas[22:26, 8:20] = 1.0
        
        elif digit == 7:
            # Top line with diagonal
            canvas[6:10, 8:20] = 1.0
            for i in range(10, 26):
                j = int(20 - (i-10) * 0.5)
                if 8 <= j < 20:
                    canvas[i, j:j+2] = 1.0
        
        elif digit == 8:
            # Two circles stacked
            canvas[6:10, 10:18] = 1.0
            canvas[10:14, 8:12] = 1.0
            canvas[10:14, 16:20] = 1.0
            canvas[14:18, 10:18] = 1.0
            canvas[18:22, 8:12] = 1.0
            canvas[18:22, 16:20] = 1.0
            canvas[22:26, 10:18] = 1.0
        
        elif digit == 9:
            # Circle with gap at bottom left
            canvas[6:10, 10:18] = 1.0
            canvas[10:14, 8:12] = 1.0
            canvas[10:14, 16:20] = 1.0
            canvas[14:18, 10:20] = 1.0
            canvas[18:26, 16:20] = 1.0
        
        return canvas
    
    def demo_braille_display(self):
        """Demonstrate braille display functionality"""
        # Create a simple test pattern
        test_image = np.zeros((28, 28))
        
        # Draw a simple smiley face
        # Eyes
        test_image[8:12, 8:12] = 1
        test_image[8:12, 16:20] = 1
        
        # Mouth
        test_image[18:22, 10:18] = 1
        
        print("BrailleUtils Demo")
        print("=================")
        
        self.display_image_braille(test_image, "Test Pattern (Smiley Face)")
        
        # Test MNIST-style digits
        print("\nMNIST-style Digit Tests:")
        print("========================")
        for digit in range(10):
            digit_image = self.generate_test_digit(digit)
            self.display_image_braille(digit_image, f"MNIST-style Digit: {digit}")
        
        # Demonstrate 2x2 digit display
        print("2x2 Digit Grid Demo:")
        print("====================")
        
        # Create simple digit patterns
        digits = [
            np.random.rand(14, 14) > 0.7,  # Random pattern
            np.eye(14),                     # Diagonal line
            np.ones((14, 14)) * 0.3,       # Light gray
            np.zeros((14, 14))             # Black
        ]
        
        grid_display = self.display_2x2_digits(digits)
        print(grid_display)
        
        # Demo training sample display
        print("\nTraining Sample Display Demo:")
        print("=============================")
        
        # Generate 4 test digit images
        sample_images = np.array([
            self.generate_test_digit(1),
            self.generate_test_digit(2), 
            self.generate_test_digit(3),
            self.generate_test_digit(4)
        ])
        
        expected_digits = [1, 2, 3, 4]
        self.display_training_samples(sample_images, expected_digits, epoch=5, batch_idx=10)

        # Demo training sample display with predictions
        print("\nTraining Sample Display Demo with Predictions:")
        print("==============================================")
        
        # Generate 4 test digit images
        sample_images = np.array([
            self.generate_test_digit(1),
            self.generate_test_digit(2), 
            self.generate_test_digit(3),
            self.generate_test_digit(4)
        ])
        
        expected_digits = [1, 2, 3, 4]
        predicted_digits = [1, 7, 3, 9]  # Some correct, some wrong
        self.display_training_samples(sample_images, expected_digits, predicted_digits, epoch=5, batch_idx=10, sample_idx=1)

        # Demo stacked image display
        print("\nStacked Image Display Demo:")
        print("===========================")
        
        # Create a 56x56 stacked image
        stacked_test = np.zeros((56, 56))
        # Add some test patterns in each quadrant
        stacked_test[:28, :28] = self.generate_test_digit(1)
        stacked_test[:28, 28:] = self.generate_test_digit(2)
        stacked_test[28:, :28] = self.generate_test_digit(3)
        stacked_test[28:, 28:] = self.generate_test_digit(4)
        
        self.display_stacked_image_sample(
            stacked_test,
            [1, 2, 3, 4],
            [1, 7, 3, 9],
            epoch=5,
            sample_idx=1
        )

# Backward compatibility - maintain old function interface
def demo_braille_display():
    """Maintain backward compatibility with old function interface"""
    braille_utils = BrailleUtils()
    braille_utils.demo_braille_display()

def test(digit: int) -> None:
    """Maintain backward compatibility with old test function"""
    braille_utils = BrailleUtils()
    digit_image = braille_utils.generate_test_digit(digit)
    braille_utils.display_image_braille(digit_image, f"MNIST-style Digit: {digit}")

if __name__ == "__main__":
    demo_braille_display()