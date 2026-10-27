def test_capture_and_tracker_config():
    from src.contracts import PipelineConfig

    config = PipelineConfig()
    assert config.capture_mode == "threaded"
    assert config.capture_pace_files is True
    assert config.tracker_type == "bytetrack"
    assert config.onnx_providers == "auto"
    assert config.tracker_track_thresh == 0.4
    assert config.tracker_match_thresh == 0.7
