# Nix Flakes Development Guide

## Core Philosophies

### NIX-FIRST
1. **Pure Builds**: All builds are reproducible and hermetic
2. **Declarative Configuration**: Infrastructure and dependencies as code
3. **Immutability**: Dependencies are locked with exact hashes
4. **Composability**: Flakes compose cleanly with other flakes
5. **Multi-Platform**: Support Linux, macOS, and cross-compilation
6. **Development Shells**: Consistent development environments across teams
7. **Portability**: Same environment on any machine with Nix
8. **Caching**: Binary caching for fast builds

### MODERN-NIX
1. **Flakes**: Use flakes for all projects (not legacy Nix expressions)
2. **Nix 2.19+**: Latest stable Nix with flakes enabled
3. **flake-parts**: Modular flake composition
4. **devenv**: Modern development environments
5. **flake-utils**: Standard utility functions
6. **treefmt-nix**: Universal code formatting
7. **nix-direnv**: Automatic environment loading
8. **GitHub Actions**: Nix-powered CI/CD

### REPRODUCIBILITY-FIRST
1. **Lock Files**: Always commit `flake.lock`
2. **Pure Evaluation**: No impure dependencies (no `<nixpkgs>`)
3. **Fixed Hashes**: All fetchurl/fetchgit with explicit hashes
4. **Pinned Inputs**: Pin nixpkgs to specific commits
5. **Sandboxed Builds**: Enable sandbox for all builds
6. **Deterministic**: Same inputs always produce same outputs
7. **Versioned**: Track flake.lock changes in version control

### HEXAGONAL-ARCHITECTURE
1. **Domain Isolation**: Core business logic independent of Nix
2. **Build Adapters**: Nix expressions as infrastructure adapters
3. **Shell Adapters**: Development shells provide tool ports
4. **Platform Abstraction**: Multi-platform builds via Nix abstraction
5. **Dependency Inversion**: Application doesn't depend on Nix details

### AGENT-VERIFICATION
When generating Nix flakes, agents MUST verify:
1. **Evaluation**: `nix flake check` succeeds
2. **Build**: `nix build` completes without errors
3. **Development Shell**: `nix develop` provides required tools
4. **Formatting**: `nix fmt` produces clean output
5. **Lock File**: `flake.lock` is valid and committed
6. **Multi-Platform**: Builds on target platforms
7. **Tests**: `nix flake check` runs all tests successfully

---

## 1. Test-Driven Development (TDD) for Flakes

### A. TDD Protocol for Flake Development

**Red-Green-Refactor Cycle**:

1. **RED**: Write failing test (e.g., check that fails)
2. **GREEN**: Implement minimal flake configuration to pass test
3. **REFACTOR**: Improve flake structure while keeping tests green
4. **VERIFY**: Run `nix flake check` to ensure all tests pass

### B. Flake Development TDD Example

**Step 1: RED - Write Failing Check**

```nix
# flake.nix (Initial - Test First)
{
  description = "My Application with TDD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        # RED: Define checks that will initially fail
        checks = {
          # Test: Python script must have required dependencies
          python-dependencies = pkgs.runCommand "check-python-deps" {
            buildInputs = [ pkgs.python3 pkgs.python3Packages.requests ];
          } ''
            # This will fail until we set up the environment correctly
            python3 -c "import requests; import json" || exit 1
            touch $out
          '';

          # Test: Development shell must have formatters
          dev-shell-formatters = pkgs.runCommand "check-formatters" {
            buildInputs = [ self.devShells.${system}.default ];
          } ''
            # This will fail until we add formatters to devShell
            command -v black || exit 1
            command -v ruff || exit 1
            touch $out
          '';

          # Test: Build output must be executable
          build-executable = pkgs.runCommand "check-executable" {
            buildInputs = [ self.packages.${system}.default ];
          } ''
            # This will fail until we create the package
            test -x ${self.packages.${system}.default}/bin/myapp || exit 1
            touch $out
          '';
        };
      }
    );
}
```

**Verify RED State**:
```bash
nix flake check
# Expected: Errors because packages and devShells don't exist yet
```

**Step 2: GREEN - Implement Minimal Solution**

```nix
# flake.nix (GREEN - Minimal Implementation)
{
  description = "My Application with TDD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.withPackages (ps: with ps; [
          requests
        ]);
      in
      {
        packages.default = pkgs.writeShellScriptBin "myapp" ''
          #!${pkgs.bash}/bin/bash
          ${python}/bin/python3 ${./src/main.py}
        '';

        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.python3Packages.black
            pkgs.ruff
          ];
        };

        checks = {
          python-dependencies = pkgs.runCommand "check-python-deps" {
            buildInputs = [ python ];
          } ''
            python3 -c "import requests; import json"
            touch $out
          '';

          dev-shell-formatters = pkgs.runCommand "check-formatters" {
            buildInputs = with pkgs; [ python3Packages.black ruff ];
          } ''
            command -v black
            command -v ruff
            touch $out
          '';

          build-executable = pkgs.runCommand "check-executable" {
            buildInputs = [ self.packages.${system}.default ];
          } ''
            test -x ${self.packages.${system}.default}/bin/myapp
            touch $out
          '';
        };
      }
    );
}
```

**Verify GREEN State**:
```bash
nix flake check
# All checks pass ✓
```

**Step 3: REFACTOR - Improve Structure**

```nix
# flake.nix (REFACTOR - Modular Structure)
{
  description = "My Application with clean architecture";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, flake-parts }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        # Extract Python environment to reusable component
        _module.args.pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          requests
          # Add more dependencies here
        ]);

        # Package definition
        packages.default = pkgs.writeShellApplication {
          name = "myapp";
          runtimeInputs = [ config._module.args.pythonEnv ];
          text = ''
            python3 ${./src/main.py} "$@"
          '';
        };

        # Development shell with all tools
        devShells.default = pkgs.mkShell {
          inputsFrom = [ self'.packages.default ];
          buildInputs = with pkgs; [
            # Python development tools
            python3Packages.black
            python3Packages.mypy
            ruff
            python3Packages.pytest

            # General development tools
            git
            just  # Command runner
            treefmt  # Universal formatter
          ];

          shellHook = ''
            echo "🚀 Development environment loaded!"
            echo "Python: $(python3 --version)"
            echo "Available commands: black, ruff, mypy, pytest"
          '';
        };

        # Comprehensive checks
        checks = {
          # Test: Python environment has required packages
          python-environment = pkgs.runCommand "check-python-env" {
            buildInputs = [ config._module.args.pythonEnv ];
          } ''
            python3 -c "import requests; import json; print('✓ Python deps OK')"
            touch $out
          '';

          # Test: Development tools are available
          dev-tools = pkgs.runCommand "check-dev-tools" {
            buildInputs = with pkgs; [
              python3Packages.black
              ruff
              python3Packages.mypy
            ];
          } ''
            command -v black >/dev/null || exit 1
            command -v ruff >/dev/null || exit 1
            command -v mypy >/dev/null || exit 1
            echo "✓ Dev tools OK"
            touch $out
          '';

          # Test: Package builds and is executable
          package-builds = pkgs.runCommand "check-package" {
            buildInputs = [ self'.packages.default ];
          } ''
            test -x ${self'.packages.default}/bin/myapp || exit 1
            echo "✓ Package OK"
            touch $out
          '';

          # Test: Package format check
          format-check = pkgs.runCommand "check-format" {
            buildInputs = [ pkgs.nixpkgs-fmt ];
          } ''
            ${pkgs.nixpkgs-fmt}/bin/nixpkgs-fmt --check ${self}
            touch $out
          '';
        };

        # Formatter for the flake itself
        formatter = pkgs.nixpkgs-fmt;
      };
    };
}
```

**Step 4: VERIFY - Run All Checks**

```bash
# Run all checks
nix flake check

# Build the package
nix build

# Test the package
./result/bin/myapp

# Enter development shell
nix develop

# Format the flake
nix fmt

# All operations succeed ✓
```

---

## 2. Bug Fix Protocol

### Every Bug Fix Requires:

1. **Reproduce**: Create a failing check that reproduces the bug
2. **Document**: Add comments in flake.nix explaining the bug
3. **Fix**: Implement the minimal fix
4. **Verify**: Ensure the new check passes
5. **Regression**: Run `nix flake check` to prevent regressions
6. **Review**: Code review focusing on the fix and check
7. **Update Lock**: Run `nix flake update` if inputs changed

### Bug Fix Example: Missing Runtime Dependency

**Bug Report**: `myapp` crashes with "command not found: curl"

**Step 1: Reproduce with Failing Check**

```nix
# flake.nix
{
  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        packages.default = pkgs.writeShellScriptBin "myapp" ''
          #!/usr/bin/env bash
          # BUG #789: Missing curl dependency causes runtime failure
          curl -s https://api.example.com/data
        '';

        checks = {
          # BUG #789: This check reproduces the runtime failure
          runtime-dependencies = pkgs.runCommand "check-runtime-deps" {
            buildInputs = [ self.packages.${system}.default ];
          } ''
            # Try to run the application
            export PATH="${self.packages.${system}.default}/bin:$PATH"
            
            # This should fail because curl is not in the runtime environment
            myapp 2>&1 | grep -q "curl: command not found" && {
              echo "ERROR: curl dependency missing in runtime"
              exit 1
            }
            
            touch $out
          '';
        };
      }
    );
}
```

**Step 2: Verify Bug Reproduction**

```bash
nix flake check
# Expected: Check fails with "curl dependency missing"
```

**Step 3: Fix the Bug**

```nix
# flake.nix (FIXED)
{
  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        packages.default = pkgs.writeShellApplication {
          name = "myapp";
          # FIX #789: Add curl to runtimeInputs
          runtimeInputs = [ pkgs.curl ];
          text = ''
            # FIX #789: curl is now available in PATH at runtime
            curl -s https://api.example.com/data
          '';
        };

        checks = {
          # BUG #789: Verify curl is available at runtime
          runtime-dependencies = pkgs.runCommand "check-runtime-deps" {
            buildInputs = [ self.packages.${system}.default ];
          } ''
            # Verify curl is in the wrapped script's PATH
            ${self.packages.${system}.default}/bin/myapp --help 2>&1 || true
            
            # Verify the wrapper includes curl
            grep -q "curl" ${self.packages.${system}.default}/bin/myapp || {
              echo "ERROR: curl not found in runtime environment"
              exit 1
            }
            
            echo "✓ Runtime dependencies OK"
            touch $out
          '';

          # Additional check: Verify all runtime dependencies are present
          all-runtime-deps = pkgs.runCommand "check-all-deps" {
            nativeBuildInputs = [ pkgs.binutils ];
          } ''
            # Check that all required binaries are in the closure
            ${pkgs.buildPackages.findutils}/bin/find \
              ${self.packages.${system}.default} \
              -type f -executable -exec file {} \; | \
              grep -q "shell script" || exit 1
            
            echo "✓ All runtime dependencies present"
            touch $out
          '';
        };
      }
    );
}
```

**Step 4: Verify Fix**

```bash
# Run checks
nix flake check
# All checks pass ✓

# Test the fixed package
nix run
# Application works correctly ✓

# Commit with bug reference
git add flake.nix flake.lock
git commit -m "fix: Add curl runtime dependency to myapp

Fixes #789

- Added curl to runtimeInputs for writeShellApplication
- Added runtime dependency check to prevent regression
- Verified all runtime dependencies are in closure"
```

### Bug Fix Example: Impure Build

**Bug Report**: Build fails on CI but works locally (impure dependency)

**Step 1: Reproduce with Failing Check**

```nix
# flake.nix (BEFORE FIX)
{
  outputs = inputs @ { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "myapp";
        src = ./.;
        
        buildPhase = ''
          # BUG #890: Impure - depends on system time
          echo "Built at: $(date)" > build-info.txt
          
          # BUG #890: Impure - depends on $HOME
          cp $HOME/.myconfig ./config 2>/dev/null || true
          
          # BUG #890: Impure - network access during build
          curl -o data.json https://example.com/data.json || true
        '';
        
        installPhase = ''
          mkdir -p $out/bin
          cp build-info.txt $out/
        '';
      };

      checks.${system} = {
        # BUG #890: Check for pure build
        pure-build = pkgs.runCommand "check-pure-build" {} ''
          # This should catch impure builds
          # Build twice and compare outputs
          out1=$(nix-build --pure -E '(import ${self}).packages.${system}.default')
          out2=$(nix-build --pure -E '(import ${self}).packages.${system}.default')
          
          diff -r $out1 $out2 || {
            echo "ERROR: Build is impure - outputs differ"
            exit 1
          }
          
          touch $out
        '';
      };
    };
}
```

**Step 2: Fix Impure Dependencies**

```nix
# flake.nix (AFTER FIX)
{
  outputs = inputs @ { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      
      # FIX #890: Fetch external data at evaluation time with fixed hash
      externalData = pkgs.fetchurl {
        url = "https://example.com/data.json";
        sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
      };
    in {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "myapp";
        src = ./.;
        
        # FIX #890: Use fixed version instead of system time
        version = "1.0.0";
        buildDate = "2026-01-18";  # Fixed at release time
        
        buildPhase = ''
          # FIX #890: Use fixed build date instead of $(date)
          echo "Version: ${version}" > build-info.txt
          echo "Build Date: ${buildDate}" >> build-info.txt
          
          # FIX #890: Use provided config, not $HOME
          ${pkgs.lib.optionalString (builtins.pathExists ./config.example) ''
            cp ${./config.example} ./config
          ''}
          
          # FIX #890: Use pre-fetched data with fixed hash
          cp ${externalData} ./data.json
        '';
        
        installPhase = ''
          mkdir -p $out/bin
          cp build-info.txt $out/
          cp data.json $out/
        '';
      };

      checks.${system} = {
        # BUG #890: Verify pure build (fixed)
        pure-build = pkgs.runCommand "check-pure-build" {
          pkg = self.packages.${system}.default;
        } ''
          # Verify no impure references
          ${pkgs.buildPackages.findutils}/bin/find $pkg -type f -exec grep -l /home {} \; && {
            echo "ERROR: Found reference to /home in output"
            exit 1
          } || true
          
          # Verify build-info.txt has fixed values
          grep -q "Version: 1.0.0" $pkg/build-info.txt || {
            echo "ERROR: Version not fixed"
            exit 1
          }
          
          echo "✓ Build is pure"
          touch $out
        '';

        # Additional: Check sandbox compliance
        sandbox-build = pkgs.runCommand "check-sandbox" {} ''
          # Verify package builds in pure sandbox
          nix build --pure-eval --restrict-eval \
            ${self}#packages.${system}.default || {
            echo "ERROR: Build fails in pure sandbox"
            exit 1
          }
          
          echo "✓ Sandbox compliant"
          touch $out
        '';
      };
    };
}
```

**Step 3: Verify Fix and Commit**

```bash
# Run checks
nix flake check --pure-eval
# All checks pass ✓

# Build in pure mode
nix build --pure-eval
# Build succeeds ✓

# Verify reproducibility
nix build --rebuild
# Output hash matches ✓

# Commit with documentation
git add flake.nix flake.lock
git commit -m "fix: Remove impure dependencies from build

Fixes #890

- Removed system time dependency (use fixed build date)
- Removed \$HOME/.myconfig dependency (use ./config.example)
- Removed network access during build (pre-fetch with fixed hash)
- Added pure-build and sandbox-build checks
- Verified build reproducibility"
```

---

## 3. Flake Structure

### A. Standard Flake Layout

```
my-project/
├── flake.nix              # Main flake definition
├── flake.lock             # Locked dependency versions (COMMIT THIS!)
├── nix/                   # Nix expressions
│   ├── packages/          # Package definitions
│   │   ├── default.nix
│   │   ├── myapp.nix
│   │   └── utils.nix
│   ├── shells/            # Development shells
│   │   ├── default.nix
│   │   ├── python.nix
│   │   └── rust.nix
│   ├── checks/            # Custom checks
│   │   ├── formatting.nix
│   │   ├── tests.nix
│   │   └── integration.nix
│   ├── modules/           # NixOS/home-manager modules
│   │   └── myservice.nix
│   └── overlays/          # Package overlays
│       └── default.nix
├── src/                   # Application source (language-specific)
│   ├── main.py
│   └── lib.py
├── tests/                 # Test files
│   └── test_main.py
├── docs/                  # Documentation
│   └── README.md
├── .envrc                 # direnv configuration
├── .gitignore
└── README.md
```

### B. Minimal Flake Template

```nix
# flake.nix (Minimal Template)
{
  description = "My project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = pkgs.callPackage ./nix/packages/default.nix { };

        devShells.default = pkgs.callPackage ./nix/shells/default.nix { };

        checks = {
          build = self.packages.${system}.default;
          format = pkgs.runCommand "check-format" { } ''
            ${pkgs.nixpkgs-fmt}/bin/nixpkgs-fmt --check ${self}
            touch $out
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
```

### C. Modular Flake with flake-parts

```nix
# flake.nix (Modular with flake-parts)
{
  description = "My project with modular flake structure";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv.url = "github:cachix/devenv";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs = inputs @ { flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # Supported systems
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      imports = [
        inputs.devenv.flakeModule
        inputs.treefmt-nix.flakeModule
        ./nix/packages
        ./nix/checks
      ];

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        # Development environment
        devenv.shells.default = {
          languages.python = {
            enable = true;
            version = "3.12";
            poetry.enable = true;
          };

          packages = with pkgs; [
            git
            just
          ];

          enterShell = ''
            echo "🚀 Development environment ready!"
          '';
        };

        # Formatting
        treefmt.config = {
          projectRootFile = "flake.nix";
          programs = {
            nixpkgs-fmt.enable = true;
            black.enable = true;
            prettier.enable = true;
          };
        };

        # Packages
        packages.default = pkgs.callPackage ./nix/packages/default.nix { };

        # Apps (runnable with `nix run`)
        apps.default = {
          type = "app";
          program = "${self'.packages.default}/bin/myapp";
        };
      };
    };
}
```

### D. Hexagonal Architecture Integration

```nix
# flake.nix (Hexagonal Architecture)
{
  description = "Application with hexagonal architecture";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Domain: Core business logic (pure)
        domain = pkgs.python3Packages.buildPythonPackage {
          pname = "myapp-domain";
          version = "1.0.0";
          src = ./src/domain;
          
          propagatedBuildInputs = with pkgs.python3Packages; [
            pydantic  # For domain models
          ];
          
          checkInputs = with pkgs.python3Packages; [
            pytest
            pytest-cov
          ];
          
          checkPhase = ''
            pytest tests/domain
          '';
        };

        # Application: Use cases and ports
        application = pkgs.python3Packages.buildPythonPackage {
          pname = "myapp-application";
          version = "1.0.0";
          src = ./src/application;
          
          propagatedBuildInputs = [
            domain
          ];
          
          checkInputs = with pkgs.python3Packages; [
            pytest
            pytest-mock
          ];
          
          checkPhase = ''
            pytest tests/application
          '';
        };

        # Infrastructure: Adapters (databases, APIs, etc.)
        infrastructure = pkgs.python3Packages.buildPythonPackage {
          pname = "myapp-infrastructure";
          version = "1.0.0";
          src = ./src/infrastructure;
          
          propagatedBuildInputs = [
            domain
            application
          ] ++ (with pkgs.python3Packages; [
            sqlalchemy  # Database adapter
            requests    # HTTP adapter
            redis       # Cache adapter
          ]);
          
          checkInputs = with pkgs.python3Packages; [
            pytest
            pytest-docker
          ];
          
          checkPhase = ''
            pytest tests/infrastructure
          '';
        };

        # Main application: Wires everything together
        myapp = pkgs.python3Packages.buildPythonApplication {
          pname = "myapp";
          version = "1.0.0";
          src = ./src;
          
          propagatedBuildInputs = [
            domain
            application
            infrastructure
          ];
          
          # Integration tests
          checkInputs = with pkgs.python3Packages; [
            pytest
            pytest-asyncio
          ];
          
          checkPhase = ''
            pytest tests/integration
          '';
        };

      in
      {
        packages = {
          default = myapp;
          domain = domain;
          application = application;
          infrastructure = infrastructure;
        };

        # Development shells for each layer
        devShells = {
          # Default: Full stack
          default = pkgs.mkShell {
            inputsFrom = [ myapp ];
            buildInputs = with pkgs; [
              python3Packages.black
              python3Packages.mypy
              python3Packages.pytest
              python3Packages.pytest-cov
            ];
          };

          # Domain-only shell (no infrastructure dependencies)
          domain = pkgs.mkShell {
            inputsFrom = [ domain ];
            buildInputs = with pkgs; [
              python3Packages.black
              python3Packages.mypy
              python3Packages.pytest
            ];
            shellHook = ''
              echo "💎 Domain layer shell - pure business logic only"
            '';
          };

          # Application shell (domain + use cases)
          application = pkgs.mkShell {
            inputsFrom = [ application ];
            buildInputs = with pkgs; [
              python3Packages.black
              python3Packages.mypy
              python3Packages.pytest
              python3Packages.pytest-mock
            ];
            shellHook = ''
              echo "🎯 Application layer shell - use cases and ports"
            '';
          };
        };

        # Checks for each layer
        checks = {
          domain-tests = pkgs.runCommand "domain-tests" {
            buildInputs = [ domain ];
          } ''
            pytest ${./tests/domain} --cov=${domain}
            touch $out
          '';

          application-tests = pkgs.runCommand "application-tests" {
            buildInputs = [ application ];
          } ''
            pytest ${./tests/application} --cov=${application}
            touch $out
          '';

          infrastructure-tests = pkgs.runCommand "infrastructure-tests" {
            buildInputs = [ infrastructure ];
          } ''
            pytest ${./tests/infrastructure}
            touch $out
          '';

          integration-tests = pkgs.runCommand "integration-tests" {
            buildInputs = [ myapp ];
          } ''
            pytest ${./tests/integration}
            touch $out
          '';

          # Verify layer dependencies (domain has no infrastructure deps)
          layer-independence = pkgs.runCommand "check-layers" { } ''
            # Domain should not depend on infrastructure
            ! grep -r "infrastructure" ${domain} || {
              echo "ERROR: Domain depends on infrastructure!"
              exit 1
            }
            
            # Application should not depend on infrastructure
            ! grep -r "infrastructure" ${application} || {
              echo "ERROR: Application depends on infrastructure!"
              exit 1
            }
            
            echo "✓ Layer independence maintained"
            touch $out
          '';
        };
      }
    );
}
```

---

## 4. Development Shells

### A. Basic Development Shell

```nix
# nix/shells/default.nix
{ pkgs }:

pkgs.mkShell {
  name = "dev-shell";

  buildInputs = with pkgs; [
    # Language runtimes
    python312
    nodejs_20
    
    # Development tools
    git
    just        # Command runner
    direnv      # Automatic environment loading
    
    # Formatters
    nixpkgs-fmt
    nodePackages.prettier
    
    # Linters
    shellcheck
    hadolint    # Dockerfile linter
  ];

  shellHook = ''
    echo "🚀 Development environment loaded!"
    echo ""
    echo "Tools available:"
    echo "  Python: $(python --version)"
    echo "  Node: $(node --version)"
    echo "  Git: $(git --version)"
    echo ""
    echo "Run 'just --list' to see available commands"
  '';
}
```

### B. Language-Specific Shells

**Python Development**:

```nix
# nix/shells/python.nix
{ pkgs }:

let
  python = pkgs.python312;
  pythonEnv = python.withPackages (ps: with ps; [
    # Core dependencies
    requests
    pydantic
    sqlalchemy
    
    # Development tools
    black
    mypy
    ruff
    pytest
    pytest-cov
    pytest-asyncio
    ipython
  ]);
in
pkgs.mkShell {
  name = "python-dev";

  buildInputs = [
    pythonEnv
    pkgs.poetry
    pkgs.python312Packages.poetry-core
  ];

  shellHook = ''
    echo "🐍 Python development environment"
    echo "Python: $(python --version)"
    echo "Poetry: $(poetry --version)"
    
    # Set up virtual environment location
    export POETRY_VIRTUALENVS_IN_PROJECT=true
    
    # Python environment variables
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
  '';
}
```

**Rust Development**:

```nix
# nix/shells/rust.nix
{ pkgs }:

pkgs.mkShell {
  name = "rust-dev";

  buildInputs = with pkgs; [
    # Rust toolchain
    rustc
    cargo
    rustfmt
    clippy
    rust-analyzer
    
    # Additional tools
    cargo-watch
    cargo-edit
    cargo-audit
    
    # System dependencies (example)
    openssl
    pkg-config
  ];

  # Required for some crates
  RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";

  shellHook = ''
    echo "🦀 Rust development environment"
    echo "Rust: $(rustc --version)"
    echo "Cargo: $(cargo --version)"
  '';
}
```

**Node.js/TypeScript Development**:

```nix
# nix/shells/nodejs.nix
{ pkgs }:

pkgs.mkShell {
  name = "nodejs-dev";

  buildInputs = with pkgs; [
    nodejs_20
    nodePackages.npm
    nodePackages.pnpm
    nodePackages.yarn
    
    # TypeScript tools
    nodePackages.typescript
    nodePackages.typescript-language-server
    
    # Linters and formatters
    nodePackages.eslint
    nodePackages.prettier
    
    # Build tools
    nodePackages.vite
  ];

  shellHook = ''
    echo "📦 Node.js development environment"
    echo "Node: $(node --version)"
    echo "npm: $(npm --version)"
    
    # Set npm prefix to local directory
    export NPM_CONFIG_PREFIX="$PWD/.npm-global"
    export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
  '';
}
```

### C. Multi-Language Monorepo Shell

```nix
# flake.nix (Monorepo)
{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells = {
          # Full monorepo shell
          default = pkgs.mkShell {
            name = "monorepo";
            
            buildInputs = with pkgs; [
              # Backend (Python)
              python312
              poetry
              
              # Frontend (Node.js/TypeScript)
              nodejs_20
              nodePackages.pnpm
              
              # Infrastructure
              terraform
              kubectl
              docker-compose
              
              # Shared tools
              git
              just
              jq
            ];
            
            shellHook = ''
              echo "🏗️  Monorepo development environment"
              echo ""
              echo "Backend (Python): $(python --version)"
              echo "Frontend (Node): $(node --version)"
              echo "Infrastructure: terraform $(terraform version -json | jq -r .terraform_version)"
            '';
          };

          # Backend-only shell
          backend = pkgs.callPackage ./nix/shells/python.nix { };

          # Frontend-only shell
          frontend = pkgs.callPackage ./nix/shells/nodejs.nix { };

          # Infrastructure-only shell
          infrastructure = pkgs.mkShell {
            buildInputs = with pkgs; [
              terraform
              kubectl
              docker-compose
              awscli2
            ];
          };
        };
      }
    );
}
```

---

## 5. Multi-Platform Support

### A. Cross-Platform Package

```nix
# flake.nix (Cross-platform)
{
  outputs = { self, nixpkgs, flake-utils }:
    # Build for all default systems
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "myapp";
          src = ./.;
          
          # Cross-platform build
          buildPhase = ''
            # Platform-specific handling
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              echo "Building for Linux"
            ''}
            ${pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
              echo "Building for macOS"
            ''}
          '';
          
          installPhase = ''
            mkdir -p $out/bin
            cp myapp $out/bin/
          '';
        };
      }
    );
}
```

### B. Platform-Specific Configuration

```nix
# flake.nix (Platform-specific)
{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Platform-specific dependencies
        platformDeps = if pkgs.stdenv.isLinux then [
          pkgs.libnotify
          pkgs.systemd
        ] else if pkgs.stdenv.isDarwin then [
          pkgs.darwin.apple_sdk.frameworks.Cocoa
          pkgs.darwin.apple_sdk.frameworks.Foundation
        ] else [ ];
        
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "myapp";
          src = ./.;
          
          buildInputs = [
            pkgs.pkg-config
          ] ++ platformDeps;
          
          # Platform-specific configuration
          configureFlags = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            "--with-macos-sdk"
          ];
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.git
          ] ++ platformDeps;
          
          shellHook = ''
            echo "Platform: ${system}"
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              echo "Linux-specific setup"
              export LINUX_SPECIFIC=1
            ''}
            ${pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
              echo "macOS-specific setup"
              export MACOS_SPECIFIC=1
            ''}
          '';
        };
      }
    );
}
```

### C. Cross-Compilation

```nix
# flake.nix (Cross-compilation)
{
  outputs = { self, nixpkgs }:
    let
      # Build on x86_64-linux for multiple targets
      buildSystem = "x86_64-linux";
      targetSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      
      # Helper to create cross-compiled packages
      mkCrossPackage = targetSystem:
        let
          pkgs = import nixpkgs {
            system = buildSystem;
            crossSystem = nixpkgs.lib.systems.elaborate targetSystem;
          };
        in
        pkgs.stdenv.mkDerivation {
          name = "myapp-${targetSystem}";
          src = ./.;
          
          buildPhase = ''
            echo "Cross-compiling for ${targetSystem}"
            $CC -o myapp main.c
          '';
          
          installPhase = ''
            mkdir -p $out/bin
            cp myapp $out/bin/
          '';
        };
      
    in
    {
      # Native packages for each system
      packages = nixpkgs.lib.genAttrs targetSystems (system: {
        default = (import nixpkgs { inherit system; }).callPackage ./. { };
      });
      
      # Cross-compiled packages
      crossPackages.${buildSystem} = nixpkgs.lib.genAttrs targetSystems (target: {
        default = mkCrossPackage target;
      });
    };
}
```

---

## 6. Testing and Checks

### A. Comprehensive Checks

```nix
# flake.nix (Comprehensive checks)
{
  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        checks = {
          # Build check
          build = self.packages.${system}.default;

          # Format check
          format = pkgs.runCommand "check-format" { } ''
            ${pkgs.nixpkgs-fmt}/bin/nixpkgs-fmt --check ${self}
            touch $out
          '';

          # Flake evaluation check
          flake-check = pkgs.runCommand "flake-check" { } ''
            cd ${self}
            ${pkgs.nix}/bin/nix flake check --all-systems
            touch $out
          '';

          # Unit tests
          unit-tests = pkgs.runCommand "unit-tests" {
            buildInputs = [ self.packages.${system}.default ];
          } ''
            pytest ${./tests/unit}
            touch $out
          '';

          # Integration tests
          integration-tests = pkgs.runCommand "integration-tests" {
            buildInputs = [
              self.packages.${system}.default
              pkgs.docker-compose
            ];
          } ''
            pytest ${./tests/integration}
            touch $out
          '';

          # Lint check
          lint = pkgs.runCommand "lint" { } ''
            ${pkgs.statix}/bin/statix check ${self}
            ${pkgs.deadnix}/bin/deadnix --fail ${self}
            touch $out
          '';

          # Security audit
          security = pkgs.runCommand "security-audit" { } ''
            # Check for vulnerable dependencies
            ${pkgs.vulnix}/bin/vulnix ${self.packages.${system}.default}
            touch $out
          '';

          # Documentation build
          docs = pkgs.runCommand "build-docs" {
            buildInputs = [ pkgs.mdbook ];
          } ''
            cd ${./docs}
            mdbook build
            cp -r book $out
          '';

          # License compliance
          licenses = pkgs.runCommand "check-licenses" { } ''
            # Ensure all dependencies have acceptable licenses
            ${pkgs.nix}/bin/nix-store -q --requisites \
              ${self.packages.${system}.default} | \
              while read path; do
                ${pkgs.nix}/bin/nix-store --query --binding license "$path" || true
              done > licenses.txt
            
            # Check for forbidden licenses (e.g., GPL in proprietary code)
            ! grep -i "gpl" licenses.txt || {
              echo "ERROR: GPL license found!"
              exit 1
            }
            
            touch $out
          '';
        };
      }
    );
}
```

### B. NixOS Integration Tests

```nix
# flake.nix (NixOS tests)
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      packages.${system}.default = pkgs.callPackage ./. { };

      checks.${system} = {
        # NixOS VM test
        nixos-test = pkgs.nixosTest {
          name = "myapp-test";
          
          nodes.machine = { config, pkgs, ... }: {
            environment.systemPackages = [ self.packages.${system}.default ];
            
            # Configure service
            systemd.services.myapp = {
              wantedBy = [ "multi-user.target" ];
              serviceConfig = {
                ExecStart = "${self.packages.${system}.default}/bin/myapp";
                Restart = "always";
              };
            };
          };
          
          testScript = ''
            start_all()
            
            # Wait for service to start
            machine.wait_for_unit("myapp.service")
            
            # Test application
            machine.succeed("myapp --version")
            
            # Test service is running
            machine.succeed("systemctl is-active myapp.service")
            
            # Test application functionality
            output = machine.succeed("curl http://localhost:8080/health")
            assert "healthy" in output
          '';
        };

        # Container test
        container-test = pkgs.runCommand "test-container" {
          buildInputs = [ pkgs.docker ];
        } ''
          # Build container
          docker load < ${self.packages.${system}.container}
          
          # Run container
          docker run --name test-myapp -d myapp:latest
          
          # Wait for startup
          sleep 5
          
          # Test container
          docker exec test-myapp myapp --version
          
          # Cleanup
          docker stop test-myapp
          docker rm test-myapp
          
          touch $out
        '';
      };
    };
}
```

---

## 7. CI/CD Integration

### A. GitHub Actions with Nix

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  nix-check:
    name: Nix Flake Check
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Nix
        uses: cachix/install-nix-action@v24
        with:
          nix_path: nixpkgs=channel:nixos-unstable
          extra_nix_config: |
            experimental-features = nix-command flakes
            access-tokens = github.com=${{ secrets.GITHUB_TOKEN }}

      - name: Setup Cachix
        uses: cachix/cachix-action@v13
        with:
          name: my-project
          authToken: ${{ secrets.CACHIX_AUTH_TOKEN }}

      - name: Check flake
        run: nix flake check --all-systems --show-trace

      - name: Build package
        run: nix build --print-build-logs

      - name: Run tests
        run: nix flake check --print-build-logs

      - name: Check formatting
        run: nix fmt -- --check .

  nix-build-matrix:
    name: Build on ${{ matrix.system }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            system: x86_64-linux
          - os: macos-latest
            system: x86_64-darwin
          - os: macos-latest
            system: aarch64-darwin

    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v24
        with:
          extra_nix_config: |
            experimental-features = nix-command flakes

      - name: Build for ${{ matrix.system }}
        run: |
          nix build .#packages.${{ matrix.system }}.default

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: myapp-${{ matrix.system }}
          path: result/bin/*

  security:
    name: Security Audit
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v24

      - name: Run security audit
        run: |
          nix build
          nix run nixpkgs#vulnix -- ./result

      - name: Check for secrets
        run: |
          nix run nixpkgs#gitleaks -- detect --source . --verbose
```

### B. GitLab CI with Nix

```yaml
# .gitlab-ci.yml
image: nixos/nix:latest

variables:
  NIX_CONFIG: "experimental-features = nix-command flakes"

stages:
  - check
  - build
  - test
  - deploy

before_script:
  - nix --version
  - nix flake show

flake-check:
  stage: check
  script:
    - nix flake check --all-systems --show-trace
  cache:
    key: nix-store
    paths:
      - /nix/store

build:
  stage: build
  script:
    - nix build --print-build-logs
    - cp -r result/ myapp/
  artifacts:
    paths:
      - myapp/
    expire_in: 1 week

test:
  stage: test
  script:
    - nix flake check --print-build-logs
  dependencies:
    - build

format-check:
  stage: check
  script:
    - nix fmt -- --check .
  allow_failure: false

deploy:
  stage: deploy
  script:
    - nix build .#container
    - docker load < result
    - docker push myregistry/myapp:latest
  only:
    - main
```

---

## 8. Prohibited Practices

### Never Do:

1. **Impure Inputs**
   ```nix
   # ❌ BAD: Using impure nixpkgs
   let pkgs = import <nixpkgs> {};
   
   # ✅ GOOD: Use flake input
   inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
   ```

2. **Uncommitted Lock Files**
   ```bash
   # ❌ BAD: Gitignore flake.lock
   echo "flake.lock" >> .gitignore
   
   # ✅ GOOD: Commit flake.lock
   git add flake.lock
   git commit -m "chore: Update flake.lock"
   ```

3. **Unfixed Hashes**
   ```nix
   # ❌ BAD: No hash for external resource
   fetchurl {
     url = "https://example.com/file.tar.gz";
   }
   
   # ✅ GOOD: Fixed hash
   fetchurl {
     url = "https://example.com/file.tar.gz";
     sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
   }
   ```

4. **Network Access in Builds**
   ```nix
   # ❌ BAD: Network access during build
   buildPhase = ''
     curl -O https://example.com/data.json
   '';
   
   # ✅ GOOD: Fetch at evaluation time
   let
     data = fetchurl {
       url = "https://example.com/data.json";
       sha256 = "...";
     };
   in
   buildPhase = ''
     cp ${data} data.json
   '';
   ```

5. **Missing System Support**
   ```nix
   # ❌ BAD: Only supports one system
   packages.x86_64-linux.default = ...;
   
   # ✅ GOOD: Use eachDefaultSystem
   flake-utils.lib.eachDefaultSystem (system: {
     packages.default = ...;
   });
   ```

6. **Skipping Checks**
   ```nix
   # ❌ BAD: No checks defined
   outputs = { self, nixpkgs }: { };
   
   # ✅ GOOD: Define comprehensive checks
   checks = {
     build = self.packages.${system}.default;
     tests = ...;
     format = ...;
   };
   ```

7. **Non-Reproducible Builds**
   ```nix
   # ❌ BAD: Using current date/time
   buildPhase = ''
     echo "Built on $(date)" > build-info.txt
   '';
   
   # ✅ GOOD: Use fixed metadata
   buildPhase = ''
     echo "Version: ${version}" > build-info.txt
   '';
   ```

8. **Hardcoded Paths**
   ```nix
   # ❌ BAD: Hardcoded absolute paths
   postInstall = ''
     ln -s /usr/bin/python $out/bin/python
   '';
   
   # ✅ GOOD: Use Nix store paths
   postInstall = ''
     ln -s ${pkgs.python3}/bin/python $out/bin/python
   '';
   ```

9. **Modifying Source Files**
   ```nix
   # ❌ BAD: Modifying source in place
   preBuild = ''
     sed -i 's/foo/bar/' src/main.c
   '';
   
   # ✅ GOOD: Use patches or substituteInPlace
   patches = [ ./fix-foo.patch ];
   # OR
   preBuild = ''
     substituteInPlace src/main.c --replace "foo" "bar"
   '';
   ```

10. **No Development Shell**
    ```nix
    # ❌ BAD: No devShell defined
    outputs = { self, nixpkgs }: {
      packages.default = ...;
    };
    
    # ✅ GOOD: Provide devShell
    devShells.default = pkgs.mkShell {
      inputsFrom = [ self.packages.default ];
      buildInputs = [ /* dev tools */ ];
    };
    ```

---

## 9. Deployment Checklist

### Before Every Release:

#### Flake Integrity
- [ ] `nix flake check` passes on all systems
- [ ] `flake.lock` is committed and up-to-date
- [ ] All inputs have specific revisions (not "latest")
- [ ] Pure evaluation mode works: `nix build --pure-eval`

#### Reproducibility
- [ ] All external resources have fixed hashes
- [ ] No network access during build phase
- [ ] Build is deterministic (same inputs = same output)
- [ ] No hardcoded paths or absolute references
- [ ] No usage of `<nixpkgs>` or impure inputs

#### Multi-Platform
- [ ] Builds on all target systems (Linux, macOS, etc.)
- [ ] Cross-compilation works (if applicable)
- [ ] Platform-specific dependencies handled correctly
- [ ] System-specific configuration tested

#### Testing
- [ ] All checks pass: `nix flake check`
- [ ] Unit tests pass in Nix environment
- [ ] Integration tests pass
- [ ] NixOS tests pass (if applicable)
- [ ] Container builds work (if applicable)

#### Development Environment
- [ ] `nix develop` provides complete dev environment
- [ ] All necessary tools available in devShell
- [ ] direnv configuration works (if using direnv)
- [ ] Multiple devShells for different use cases (if needed)

#### Code Quality
- [ ] Formatted: `nix fmt` produces no changes
- [ ] No dead code: `deadnix --fail .`
- [ ] No anti-patterns: `statix check .`
- [ ] Documentation is complete and accurate

#### Security
- [ ] No vulnerable dependencies: `vulnix result`
- [ ] No secrets in flake.nix or committed files
- [ ] License compliance checked
- [ ] Security audit passed

#### Hexagonal Architecture
- [ ] Domain layer builds independently
- [ ] Application layer doesn't depend on infrastructure
- [ ] Infrastructure adapters properly isolated
- [ ] Layer independence checks pass

#### CI/CD
- [ ] GitHub Actions / GitLab CI passes
- [ ] Cachix caching configured (if applicable)
- [ ] Build artifacts uploaded
- [ ] Container images pushed (if applicable)

#### Documentation
- [ ] README updated with nix commands
- [ ] Flake outputs documented
- [ ] Development setup instructions clear
- [ ] Troubleshooting guide updated

#### Version Control
- [ ] All changes committed
- [ ] `flake.lock` committed
- [ ] Commit messages follow Conventional Commits
- [ ] Version number updated
- [ ] Git tag created

---

## 10. Why This Configuration Works

### Reproducibility
- **Locked Dependencies**: `flake.lock` ensures exact versions
- **Pure Evaluation**: No impure dependencies or network access
- **Sandboxed Builds**: Isolated build environment
- **Fixed Hashes**: All external resources verified

### Portability
- **Multi-Platform**: Single flake works on Linux, macOS, etc.
- **Consistent Environments**: Same tools everywhere
- **No System Dependencies**: Everything provided by Nix
- **Cross-Compilation**: Build for any target from any host

### Development Velocity
- **Fast Iteration**: Binary caching (Cachix)
- **Instant Environments**: `nix develop` in seconds
- **Automatic Loading**: direnv integration
- **Multiple Shells**: Different environments for different tasks

### Maintainability
- **Declarative**: Infrastructure as code
- **Composable**: Flakes compose cleanly
- **Modular**: flake-parts for organization
- **Versioned**: Lock file tracks all dependencies

### Hexagonal Architecture
- **Layer Isolation**: Each layer can build independently
- **Dependency Inversion**: Infrastructure doesn't leak into domain
- **Testability**: Easy to test layers in isolation
- **Flexibility**: Swap infrastructure without changing domain

### Testing
- **Comprehensive Checks**: `nix flake check` runs all tests
- **VM Testing**: NixOS tests for integration testing
- **Container Testing**: Test Docker images
- **Multi-System**: Test on all platforms

### Security
- **Vulnerability Scanning**: vulnix integration
- **License Compliance**: Automated checking
- **Audit Trail**: Lock file provides complete dependency tree
- **Immutability**: Can't accidentally modify dependencies

This configuration creates a solid foundation for building portable,
reproducible development environments and applications with Nix flakes.

---

## Additional Resources

- [Nix Flakes Documentation](https://nixos.wiki/wiki/Flakes)
- [Nix Pills](https://nixos.org/guides/nix-pills/)
- [Zero to Nix](https://zero-to-nix.com/)
- [nix.dev](https://nix.dev/)
- [flake-parts](https://flake.parts/)
- [devenv](https://devenv.sh/)
- [nixpkgs Manual](https://nixos.org/manual/nixpkgs/stable/)
- [Nix Package Versions](https://lazamar.co.uk/nix-versions/)
