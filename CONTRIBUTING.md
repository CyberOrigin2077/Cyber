# 🚀 Contributing to CYBER

Thanks for your interest in contributing to CYBER! We welcome contributions of all kinds, from bug fixes to new features.

## 🌟 Ways to Contribute

### 1. 🐛 Report Bugs & Issues
- Use our bug report template when opening issues
- Include clear reproduction steps
- Provide environment details (OS, Python version, GPU)
- Add relevant logs or screenshots
- Tag issues appropriately (bug, documentation, enhancement, etc.)

### 2. 💡 Share Ideas & Discussions
- Join our [Discussions](https://github.com/CyberOrigin2077/Cyber/discussions) for:
  - Feature proposals
  - Architecture discussions
  - Best practices
  - Use cases & applications
  - Questions & answers

### 3. 🔧 Submit Code Changes
- Bug fixes
- Performance improvements
- Documentation updates
- New features
- Test coverage improvements

### 4. 🤖 Contribute Models
- Share new model architectures
- Add model implementations
- Contribute pre-trained checkpoints
- Improve existing models
- Add model benchmarks

### 5. 📊 Share Datasets
- Contribute new datasets
- Add data processing scripts
- Improve data loading
- Share data augmentation techniques
- Add dataset documentation

### 6. 📚 Improve Documentation
- Fix typos and clarify explanations
- Add code examples
- Write tutorials
- Create diagrams
- Translate documentation

## 💬 Getting Help

- Open a [Discussion](https://github.com/CyberOrigin2077/Cyber/discussions)
- Email: contact@cyberorigin.ai
- Check existing issues and discussions

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## 🛠️ Development Setup Guide

### Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cyber.git
   cd Cyber
   ```

2. **Set up Poetry Environment**
   ```bash
   # Install poetry if you haven't (see https://python-poetry.org/docs/)
   # IMPORTANT: Poetry should be installed in its own isolated environment, NOT in your project's environment.

   # Install dependencies
   poetry install

   # Activate the virtual environment
   poetry shell
   ```

3. **Install Pre-commit Hooks**
   ```bash
   poetry run pre-commit install
   ```

### Code Standards

We use several tools to maintain code quality:

1. **Ruff** for linting and formatting:
   - Line length: 160 characters
   - Python version: 3.10+
   - Auto-fixes available for many issues
   - Run: `poetry run ruff check .`

2. **MyPy** for type checking:
   - Strict optional checking
   - Run: `poetry run mypy .`

3. **Pre-commit Hooks** check:
   - Formatting (ruff-format)
   - Linting (ruff)
   - Type checking (mypy)
   - Run manually: `poetry run pre-commit run --all-files`

### PR Workflow

We follow the [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) for all contributions. Here's a detailed breakdown:

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   - Use descriptive branch names (e.g., `add-vision-model`, `fix-memory-leak`)
   - Create a separate branch for each set of unrelated changes

2. **Make Changes**
   - Write clear, documented code
   - Add tests for new features
   - Update documentation if needed
   - Commit and push changes regularly to backup your work
   - Each commit should contain an isolated, complete change

3. **Quality Checks**
   ```bash
   # Run all pre-commit hooks
   poetry run pre-commit run --all-files

   # Run tests
   poetry run pytest tests/
   ```
   - All CI checks must pass
   - Code must follow our style guidelines
   - Tests must pass

4. **Create Pull Request**
   - Write a clear PR description explaining the changes
   - Link related issues using keywords (e.g., "Fixes #123")
   - Include screenshots or examples if relevant
   - Mark as draft if you want early feedback
   - Request reviews from relevant team members

5. **Address Reviews**
   - Respond to all review comments
   - Make requested changes
   - Push additional commits as needed
   - Get required approvals

6. **Merge**
   - PR must have required approvals
   - All CI checks must pass
   - No merge conflicts
   - Squash commits if requested by reviewers

7. **Clean Up**
   - Delete your branch after merging
   - Close related issues if not auto-closed

For more detailed information about the GitHub Flow process, please refer to the [GitHub Flow documentation](https://docs.github.com/en/get-started/using-github/github-flow).

### Best Practices

- **Code Organization**
  - Keep files focused and modular
  - Use appropriate directory structure
  - Follow existing patterns

- **Documentation**
  - Document new functions/classes
  - Update README if needed
  - Add docstrings in NumPy format

- **Testing**
  - Write unit tests for new features
  - Ensure existing tests pass
  - Add integration tests if needed

- **PR Size**
  - Keep PRs focused on single changes
  - Split large changes into smaller PRs
  - Aim for reviewable chunks
