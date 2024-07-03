from PIL import Image
from abc import ABC, abstractmethod
import numpy as np


# Abstract base class defining the interface for palette application strategies
class PaletteApplicationStrategy(ABC):
    @abstractmethod
    def apply_palette(self, img: Image.Image, palette: list) -> Image.Image:
        """Apply a color palette to an image. """
        pass
    
    @classmethod
    def get_strategy_names(cls):
        return ['InterpolatedPaletteStrategy', 'DirectPaletteStrategy', 'PosterizationStrategy']

    @classmethod
    def create_strategy(cls, strategy_name: str):
        strategies = {
            'InterpolatedPaletteStrategy': InterpolatedPaletteStrategy,
            'DirectPaletteStrategy': DirectPaletteStrategy,
            'PosterizationStrategy': PosterizationStrategy
        }
        return strategies.get(strategy_name, DirectPaletteStrategy)()


# Strategy 1: Applies a color palette using an interpolated gradient
class InterpolatedPaletteStrategy(PaletteApplicationStrategy):
    def apply_palette(self, img: Image.Image, palette: list) -> Image.Image:
        if not palette:
            return img

        grayscale_np = self.convert_to_grayscale(img)
        gradient_map = self.create_gradient_map(palette)
        colored_img_np = self.map_grayscale_to_palette(grayscale_np, gradient_map)

        return Image.fromarray(colored_img_np)

    def convert_to_grayscale(self, img: Image.Image) -> np.ndarray:
        """Convert image to grayscale and return as numpy array."""
        grayscale = img.convert("L")
        return np.array(grayscale)

    def create_gradient_map(self, palette: list) -> np.ndarray:
        """Create a smooth gradient map from the palette."""
        palette_np = np.array(palette, dtype=np.uint8)
        gradient_map = np.zeros((256, 3), dtype=np.uint8)

        # Interpolate the palette colors across the grayscale range
        for i in range(3):  # Iterate over R, G, B channels
            gradient_map[:, i] = np.interp(
                np.arange(256),
                np.linspace(0, 255, len(palette)),
                palette_np[:, i]
            )
        return gradient_map

    def map_grayscale_to_palette(self, grayscale_np: np.ndarray, gradient_map: np.ndarray) -> np.ndarray:
        """Map grayscale values to the new palette using the gradient map."""
        return gradient_map[grayscale_np]


# Strategy 2: Applies a color palette using direct mapping
class DirectPaletteStrategy(PaletteApplicationStrategy):
    def apply_palette(self, img: Image.Image, palette: list) -> Image.Image:
        if not palette:
            return img

        # Flatten the palette list for use with the putpalette method
        flat_palette = [value for color in palette for value in color]
        
        # Convert image to palette mode using the adaptive palette
        palette_image = img.convert("P", palette=Image.ADAPTIVE, colors=len(palette))
        palette_image.putpalette(flat_palette)

        return palette_image


class PosterizationStrategy(PaletteApplicationStrategy):
    def apply_palette(self, img: Image.Image, palette: list) -> Image.Image:
        """Apply a color palette to the image using posterization method."""
        if not palette:
            return img

        grayscale_np = self.convert_to_grayscale(img)
        quantized_img_np = self.quantize_image(grayscale_np, len(palette))
        colored_img_np = self.map_quantized_to_palette(quantized_img_np, palette)

        return Image.fromarray(colored_img_np)

    def convert_to_grayscale(self, img: Image.Image) -> np.ndarray:
        """Convert image to grayscale and return as numpy array."""
        grayscale = img.convert("L")
        return np.array(grayscale)

    def quantize_image(self, grayscale_np: np.ndarray, num_colors: int) -> np.ndarray:
        bins = np.linspace(0, 256, num_colors + 1)
        quantized_img = np.digitize(grayscale_np, bins) - 1
        return quantized_img

    def map_quantized_to_palette(self, quantized_img_np: np.ndarray, palette: list) -> np.ndarray:
        palette_np = np.array(palette, dtype=np.uint8)
        colored_img_np = palette_np[quantized_img_np]
        return colored_img_np
        

# Context class to use the palette application strategies
class PaletteApplier:
    def __init__(self, strategy: PaletteApplicationStrategy):
        """Initialize with a given strategy."""
        self._strategy = strategy

    def set_strategy(self, strategy: PaletteApplicationStrategy):
        """Set a different palette application strategy."""
        self._strategy = strategy

    def apply_palette(self, img: Image.Image, palette: list) -> Image.Image:
        """Apply the current strategy's palette to the image."""
        return self._strategy.apply_palette(img, palette)
