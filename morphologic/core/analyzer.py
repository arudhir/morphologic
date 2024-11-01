class CellAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.channels = ChannelMap
        self.nucleus_analyzer = NucleusAnalyzer(config)
        self.mito_analyzer = MitochondriaAnalyzer(config)
        self.golgi_analyzer = GolgiAnalyzer(config)
        self.membrane_analyzer = MembraneAnalyzer(config)

    def process_image(self, image_paths: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process multi-channel fluorescence images for cell analysis.
        
        Channel information:
        - Channel 0 (w1): Hoechst 33342 staining for nucleus
        - Channel 1 (w2): Concanavalin A staining for membrane glycoproteins
        - Channel 2 (w3): Phalloidin staining for Actin
        - Channel 3 (w4): Syto14 staining for RNA
        - Channel 4 (w5): Wheat germ agglutinin staining for Golgi
        - Channel 5 (w6): Mitotracker staining for mitochondria
        """
        try:
            # Load and normalize images
            channels = [self.normalize_channel(io.imread(path)) for path in image_paths]
            img = np.stack(channels, axis=-1)
            print(f"Images loaded and normalized with shape: {img.shape}")
            
            # Segment using organelle analyzers with proper channels
            nuclei_masks = self.nucleus_analyzer.segment(
                img[:, :, self.channels.NUCLEUS.value.index])
            
            cell_masks = self.membrane_analyzer.segment(
                img[:, :, self.channels.ACTIN.value.index])  # Using Actin for cell boundaries
            
            mito_masks = self.mito_analyzer.segment(
                img[:, :, self.channels.MITOCHONDRIA.value.index])
            
            golgi_masks = self.golgi_analyzer.segment(
                img[:, :, self.channels.GOLGI.value.index])
            
            # Extract features with proper channel references
            nuclear_features = self.nucleus_analyzer.extract_features(
                nuclei_masks, 
                img[:, :, self.channels.NUCLEUS.value.index]
            )
            
            cell_features = self.membrane_analyzer.extract_features(
                cell_masks, 
                img[:, :, self.channels.ACTIN.value.index]
            )
            
            mito_features = self.mito_analyzer.extract_features(
                mito_masks, 
                img[:, :, self.channels.MITOCHONDRIA.value.index]
            )
            
            golgi_features = self.golgi_analyzer.extract_features(
                golgi_masks, 
                img[:, :, self.channels.GOLGI.value.index]
            )
            
            # Rest of the processing remains the same...
            
        except Exception as e:
            print(f"An error occurred during image processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    def get_channel_info(self) -> str:
        """Return formatted information about imaging channels."""
        info = "Channel Information:\n"
        for channel in ChannelMap:
            ch = channel.value
            info += f"Channel {ch.index} ({ch.name}): {ch.stain} staining for {ch.target}\n"
        return info

    # You might also want to add channel validation:
    def validate_channels(self, image_paths: List[str]) -> bool:
        """Validate that we have all required channels."""
        if len(image_paths) != len(ChannelMap):
            print(f"Error: Expected {len(ChannelMap)} channels, got {len(image_paths)}")
            return False
        
        # Add more validation as needed (e.g., check file names, dimensions)
        return True
