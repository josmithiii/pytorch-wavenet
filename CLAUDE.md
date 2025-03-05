# PyTorch WaveNet Guidelines

## Commands
- Run all tests: `python -m unittest discover tests`
- Run single test: `python -m unittest tests/test_file.py`
- Run specific test case: `python -m unittest tests/test_file.py.TestClass.test_method`
- Train model: `python demo.py --mode train`
- Resume training: `python demo.py --mode train --resume snapshots/model_name`
- Generate audio: `python demo.py --mode generate`
- View TensorBoard logs: `tensorboard --logdir=runs`

## Code Style
- **Imports**: stdlib → external → local modules
- **Naming**:
  - Classes: CamelCase (WaveNetModel)
  - Functions/variables: snake_case (generate_audio, skip_channels)
  - Constants: UPPER_SNAKE_CASE
- **Types**: Add docstrings with parameter types (NumPy style)
- **Error handling**: Use standard exceptions with descriptive messages
- **Formatting**: ~80 character line limit
- **Testing**: Use unittest.TestCase with test_ prefixed methods
- **Modules**: Model components in wavenet_modules.py, model architecture in wavenet_model.py
- **Directory structure**: Code in root, tests in tests/, checkpoints in snapshots/, logs in runs/