import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

class ImageProcessor:
    """Basic image processing for material identification"""
    
    def __init__(self):
        # Color profiles for different materials
        self.material_color_profiles = {
            'plastic': {
                'clear': [(240, 240, 240, 255), (255, 255, 255, 255)],  # Clear/white plastic
                'colored': [(0, 100, 200, 255), (50, 150, 255, 255)],   # Colored plastic bottles
                'black': [(0, 0, 0, 255), (50, 50, 50, 255)]            # Black plastic
            },
            'metal': {
                'aluminum': [(180, 180, 180, 255), (220, 220, 220, 255)],  # Aluminum silver
                'steel': [(100, 100, 100, 255), (150, 150, 150, 255)],     # Steel gray
                'copper': [(184, 115, 51, 255), (205, 127, 50, 255)]       # Copper brown
            },
            'glass': {
                'clear': [(245, 245, 255, 255), (255, 255, 255, 255)],     # Clear glass
                'brown': [(101, 67, 33, 255), (139, 90, 43, 255)],         # Brown bottles
                'green': [(34, 139, 34, 255), (50, 205, 50, 255)]          # Green glass
            },
            'paper': {
                'white': [(240, 240, 230, 255), (255, 255, 255, 255)],     # White paper
                'brown': [(139, 115, 85, 255), (160, 130, 98, 255)],       # Cardboard
                'newsprint': [(220, 220, 200, 255), (240, 240, 220, 255)]  # Newspaper
            }
        }
        
        # Texture patterns (simplified for basic detection)
        self.texture_indicators = {
            'plastic': ['smooth', 'glossy', 'flexible'],
            'metal': ['reflective', 'hard', 'metallic'],
            'paper': ['fibrous', 'matte', 'textured'],
            'glass': ['smooth', 'transparent', 'brittle'],
            'textile': ['woven', 'soft', 'flexible'],
            'electronics': ['complex', 'multi-material', 'components']
        }
    
    def classify_image(self, image):
        """Classify material type from image using basic image analysis"""
        if image is None:
            return "unknown"
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for consistent analysis
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Extract features
        color_features = self._extract_color_features(image)
        texture_features = self._extract_texture_features(image)
        shape_features = self._extract_shape_features(image)
        
        # Combine features for classification
        material_scores = self._calculate_material_scores(color_features, texture_features, shape_features)
        
        # Return the material with highest score
        best_material = max(material_scores.items(), key=lambda x: x[1])
        
        return best_material[0] if best_material[1] > 0.3 else "unknown"
    
    def _extract_color_features(self, image):
        """Extract color-based features from image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate color statistics
        features = {
            'mean_rgb': np.mean(img_array, axis=(0, 1)),
            'std_rgb': np.std(img_array, axis=(0, 1)),
            'dominant_colors': self._get_dominant_colors(img_array)
        }
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture-based features using basic image processing"""
        # Convert to grayscale
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        # Calculate texture metrics
        features = {
            'variance': np.var(gray_array),
            'gradient_magnitude': self._calculate_gradient_magnitude(gray_array),
            'edge_density': self._calculate_edge_density(gray_array)
        }
        
        return features
    
    def _extract_shape_features(self, image):
        """Extract basic shape features"""
        # Convert to grayscale and apply threshold
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        # Simple edge detection
        edges = self._simple_edge_detection(gray_array)
        
        features = {
            'edge_count': np.sum(edges > 0),
            'contour_complexity': self._calculate_contour_complexity(edges),
            'aspect_ratio': image.width / image.height
        }
        
        return features
    
    def _get_dominant_colors(self, img_array, k=3):
        """Get dominant colors using simple clustering"""
        # Flatten image to list of colors
        pixels = img_array.reshape(-1, 3)
        
        # Simple binning approach to find dominant colors
        binned = pixels // 32 * 32  # Bin colors to reduce complexity
        
        # Convert to tuples for easier counting
        color_tuples = [tuple(pixel) for pixel in binned]
        
        # Count occurrences of each color
        from collections import Counter
        color_counts = Counter(color_tuples)
        
        # Get the k most common colors
        most_common = color_counts.most_common(k)
        dominant_colors = [color for color, count in most_common]
        
        return dominant_colors
    
    def _calculate_gradient_magnitude(self, gray_array):
        """Calculate gradient magnitude for texture analysis"""
        # Simple gradient calculation
        gx = np.abs(np.diff(gray_array, axis=1))
        gy = np.abs(np.diff(gray_array, axis=0))
        
        # Pad to maintain shape
        gx = np.pad(gx, ((0, 0), (1, 0)), mode='edge')
        gy = np.pad(gy, ((1, 0), (0, 0)), mode='edge')
        
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        return np.mean(gradient_magnitude)
    
    def _calculate_edge_density(self, gray_array):
        """Calculate edge density in the image"""
        edges = self._simple_edge_detection(gray_array)
        return np.sum(edges > 0) / edges.size
    
    def _simple_edge_detection(self, gray_array):
        """Simple edge detection using gradient"""
        gx = np.abs(np.diff(gray_array, axis=1))
        gy = np.abs(np.diff(gray_array, axis=0))
        
        # Pad arrays
        gx = np.pad(gx, ((0, 0), (1, 0)), mode='constant')
        gy = np.pad(gy, ((1, 0), (0, 0)), mode='constant')
        
        edges = np.sqrt(gx**2 + gy**2)
        
        # Threshold to get binary edges
        threshold = np.mean(edges) + np.std(edges)
        return (edges > threshold).astype(np.uint8) * 255
    
    def _calculate_contour_complexity(self, edges):
        """Calculate complexity of contours in edge image"""
        # Simple measure based on edge transitions
        horizontal_transitions = np.sum(np.abs(np.diff(edges, axis=1)) > 0)
        vertical_transitions = np.sum(np.abs(np.diff(edges, axis=0)) > 0)
        
        total_transitions = horizontal_transitions + vertical_transitions
        return total_transitions / edges.size
    
    def _calculate_material_scores(self, color_features, texture_features, shape_features):
        """Calculate likelihood scores for each material type"""
        scores = {}
        
        # Plastic detection
        scores['plastic'] = self._score_plastic(color_features, texture_features, shape_features)
        
        # Metal detection
        scores['metal'] = self._score_metal(color_features, texture_features, shape_features)
        
        # Paper detection
        scores['paper'] = self._score_paper(color_features, texture_features, shape_features)
        
        # Glass detection
        scores['glass'] = self._score_glass(color_features, texture_features, shape_features)
        
        # Electronics detection
        scores['electronics'] = self._score_electronics(color_features, texture_features, shape_features)
        
        # Textile detection
        scores['textile'] = self._score_textile(color_features, texture_features, shape_features)
        
        return scores
    
    def _score_plastic(self, color_features, texture_features, shape_features):
        """Score for plastic material"""
        score = 0.0
        
        # Color indicators
        mean_rgb = color_features['mean_rgb']
        if np.mean(mean_rgb) > 200:  # Bright colors often indicate plastic
            score += 0.3
        
        # Texture indicators (smooth, low variance)
        if texture_features['variance'] < 500:  # Smooth surface
            score += 0.3
        
        # Edge density (plastic often has clean edges)
        if texture_features['edge_density'] < 0.1:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_metal(self, color_features, texture_features, shape_features):
        """Score for metal material"""
        score = 0.0
        
        # Color indicators (gray, silver tones)
        mean_rgb = color_features['mean_rgb']
        rgb_std = np.std(mean_rgb)
        rgb_brightness = np.mean(mean_rgb)
        
        # Metallic colors are typically in mid-range brightness with low color variation
        if rgb_std < 20 and 80 < rgb_brightness < 200:
            score += 0.4
        
        # High gradient magnitude indicates reflective surface
        if texture_features['gradient_magnitude'] > 20:
            score += 0.3
        
        # Pure metal typically has lower contour complexity than electronics
        if shape_features['contour_complexity'] < 0.008:
            score += 0.2
        
        # Reduce score if image has characteristics of electronics
        # (high contrast, dark areas, complex shapes)
        if (rgb_brightness < 100 or 
            shape_features['contour_complexity'] > 0.01 or 
            texture_features['edge_density'] > 0.12):
            score *= 0.6  # Reduce metal score if looks like electronics
        
        return min(1.0, score)
    
    def _score_paper(self, color_features, texture_features, shape_features):
        """Score for paper material"""
        score = 0.0
        
        # Color indicators (white, brown, beige)
        mean_rgb = color_features['mean_rgb']
        if (mean_rgb[0] > 180 and mean_rgb[1] > 180 and mean_rgb[2] > 160) or \
           (mean_rgb[0] > 100 and mean_rgb[1] > 80 and mean_rgb[2] > 60):
            score += 0.4
        
        # Texture variance (paper has fiber texture)
        if 100 < texture_features['variance'] < 1000:
            score += 0.3
        
        return min(1.0, score)
    
    def _score_glass(self, color_features, texture_features, shape_features):
        """Score for glass material"""
        score = 0.0
        
        # Glass is often very smooth with high reflectivity
        if texture_features['variance'] < 200:
            score += 0.3
        
        # High gradient magnitude from reflections
        if texture_features['gradient_magnitude'] > 15:
            score += 0.3
        
        # Color characteristics
        mean_rgb = color_features['mean_rgb']
        if np.mean(mean_rgb) > 220:  # Clear glass
            score += 0.2
        
        return min(1.0, score)
    
    def _score_electronics(self, color_features, texture_features, shape_features):
        """Score for electronics material"""
        score = 0.0
        
        # Electronics often have dark areas (screens) with metallic frames
        mean_rgb = color_features['mean_rgb']
        rgb_brightness = np.mean(mean_rgb)
        
        # Look for dark screens (common in phones/electronics)
        if rgb_brightness < 120:
            score += 0.4
        
        # High contour complexity indicates multiple components (lowered threshold)
        if shape_features['contour_complexity'] > 0.005:
            score += 0.3
        
        # Moderate to high edge density from device shapes and components
        if texture_features['edge_density'] > 0.08:
            score += 0.3
        
        # Look for rectangular aspect ratios common in devices
        aspect_ratio = shape_features['aspect_ratio']
        if 0.4 < aspect_ratio < 2.5:  # Typical phone/tablet ratios
            score += 0.2
        
        # Electronics often have high contrast (dark screens + bright frames)
        if texture_features['gradient_magnitude'] > 12:
            score += 0.3
        
        # Color variety from different materials and components
        if color_features['std_rgb'].mean() > 25:
            score += 0.2
        
        # Boost score if multiple electronic indicators are present
        indicator_count = sum([
            rgb_brightness < 120,
            shape_features['contour_complexity'] > 0.005,
            texture_features['edge_density'] > 0.08,
            0.4 < aspect_ratio < 2.5,
            texture_features['gradient_magnitude'] > 12
        ])
        
        if indicator_count >= 3:
            score += 0.3
        
        return min(1.0, score)
    
    def _score_textile(self, color_features, texture_features, shape_features):
        """Score for textile material"""
        score = 0.0
        
        # Moderate texture variance from fabric weave
        if 200 < texture_features['variance'] < 2000:
            score += 0.4
        
        # Moderate edge density from fabric patterns
        if 0.05 < texture_features['edge_density'] < 0.2:
            score += 0.3
        
        return min(1.0, score)
    
    def enhance_image(self, image):
        """Enhance image for better classification"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
        
        return image
    
    def get_image_analysis_report(self, image):
        """Get detailed analysis report of the image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        color_features = self._extract_color_features(image)
        texture_features = self._extract_texture_features(image)
        shape_features = self._extract_shape_features(image)
        
        material_scores = self._calculate_material_scores(color_features, texture_features, shape_features)
        
        report = {
            'image_size': image.size,
            'dominant_colors': color_features['dominant_colors'],
            'color_statistics': {
                'mean_rgb': color_features['mean_rgb'].tolist(),
                'std_rgb': color_features['std_rgb'].tolist()
            },
            'texture_metrics': {
                'variance': float(texture_features['variance']),
                'gradient_magnitude': float(texture_features['gradient_magnitude']),
                'edge_density': float(texture_features['edge_density'])
            },
            'shape_metrics': {
                'edge_count': int(shape_features['edge_count']),
                'contour_complexity': float(shape_features['contour_complexity']),
                'aspect_ratio': float(shape_features['aspect_ratio'])
            },
            'material_probabilities': material_scores,
            'classification': max(material_scores.items(), key=lambda x: x[1])[0]
        }
        
        return report
