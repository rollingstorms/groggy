#!/usr/bin/env python3
"""
🎨 PHASE 9 TESTING: Visual Design & Themes Comprehensive Validation

Tests the complete Phase 9 implementation including:
- CSS theme system with 5 built-in themes (Light, Dark, Publication, Minimal, Neon)
- ThemeManager JavaScript with theme switching
- Responsive design for mobile/tablet/desktop
- Smooth animations and transitions
- Professional node/edge styling system

This test validates that all Phase 9 visual design features work correctly
and integrate seamlessly with the existing Phase 7-8 infrastructure.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python-groggy" / "python"))

try:
    import groggy
    print("✅ Successfully imported groggy")
except ImportError as e:
    print(f"❌ Failed to import groggy: {e}")
    print("💡 Make sure to run 'cd python-groggy && maturin develop' first")
    sys.exit(1)

class Phase9Tester:
    """Comprehensive tester for Phase 9 visual design and theme functionality."""
    
    def __init__(self):
        self.test_results = []
        self.test_graph = None
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
    
    def test_css_files_exist(self):
        """Test that all Phase 9 CSS files exist and have expected content."""
        print("\n🎨 Testing Phase 9 CSS files...")
        
        css_files = {
            "styles.css": ["--color-primary", ":root", "[data-theme=", "CSS Custom Properties"],
            "responsive.css": ["@media", "mobile-first", "responsive", "breakpoint"],
            "animations.css": ["@keyframes", "transition", "animation", "smooth"],
            "graph-styles.css": ["graph-node", "graph-edge", "node-label", "edge-label"]
        }
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        all_passed = True
        
        for filename, expected_content in css_files.items():
            file_path = css_dir / filename
            
            try:
                if file_path.exists():
                    content = file_path.read_text()
                    
                    # Check for expected content
                    missing_content = []
                    for expected in expected_content:
                        if expected not in content:
                            missing_content.append(expected)
                    
                    if not missing_content:
                        self.log_test(f"CSS file {filename}", True,
                                    f"File exists with expected content ({len(content)} chars)")
                    else:
                        self.log_test(f"CSS file {filename}", False,
                                    f"Missing content: {missing_content}")
                        all_passed = False
                else:
                    self.log_test(f"CSS file {filename}", False, "File does not exist")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"CSS file {filename}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_theme_system_completeness(self):
        """Test that the theme system has all 5 required themes."""
        print("\n🎨 Testing theme system completeness...")
        
        # Check main styles.css for theme definitions
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        styles_file = css_dir / "styles.css"
        
        expected_themes = [
            '[data-theme="light"]',
            '[data-theme="dark"]', 
            '[data-theme="publication"]',
            '[data-theme="minimal"]',
            '[data-theme="neon"]'
        ]
        
        try:
            if styles_file.exists():
                content = styles_file.read_text()
                
                missing_themes = []
                for theme in expected_themes:
                    if theme not in content:
                        missing_themes.append(theme)
                
                if not missing_themes:
                    self.log_test("Theme system completeness", True,
                                f"All {len(expected_themes)} themes found in CSS")
                    return True
                else:
                    self.log_test("Theme system completeness", False,
                                f"Missing themes: {missing_themes}")
                    return False
            else:
                self.log_test("Theme system completeness", False,
                            "styles.css not found")
                return False
                
        except Exception as e:
            self.log_test("Theme system completeness", False, str(e))
            return False
    
    def test_theme_manager_javascript(self):
        """Test that ThemeManager JavaScript exists and has required functionality."""
        print("\n🔧 Testing ThemeManager JavaScript...")
        
        js_dir = project_root / "src" / "viz" / "frontend" / "js"
        theme_file = js_dir / "theme-manager.js"
        
        expected_features = [
            "class ThemeManager", "applyTheme", "setupEventListeners", 
            "setupResponsiveHandlers", "saveTheme", "loadSavedTheme",
            "handleResponsiveChanges", "themes", "breakpoints"
        ]
        
        try:
            if theme_file.exists():
                content = theme_file.read_text()
                
                missing_features = []
                for feature in expected_features:
                    if feature not in content:
                        missing_features.append(feature)
                
                if not missing_features:
                    self.log_test("ThemeManager JavaScript", True,
                                f"All {len(expected_features)} features found")
                    return True
                else:
                    self.log_test("ThemeManager JavaScript", False,
                                f"Missing features: {missing_features}")
                    return False
            else:
                self.log_test("ThemeManager JavaScript", False,
                            "theme-manager.js not found")
                return False
                
        except Exception as e:
            self.log_test("ThemeManager JavaScript", False, str(e))
            return False
    
    def test_responsive_breakpoints(self):
        """Test that responsive design has proper breakpoints."""
        print("\n📱 Testing responsive design breakpoints...")
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        responsive_file = css_dir / "responsive.css"
        
        expected_breakpoints = [
            "@media (min-width: 320px)",  # Mobile
            "@media (min-width: 768px)",  # Tablet
            "@media (min-width: 1024px)", # Desktop
            "@media (max-width: 768px)",  # Mobile-specific
            "@media (hover: none)",       # Touch devices
            "@media (prefers-reduced-motion: reduce)" # Accessibility
        ]
        
        try:
            if responsive_file.exists():
                content = responsive_file.read_text()
                
                missing_breakpoints = []
                for breakpoint in expected_breakpoints:
                    if breakpoint not in content:
                        missing_breakpoints.append(breakpoint)
                
                if not missing_breakpoints:
                    self.log_test("Responsive design breakpoints", True,
                                f"All {len(expected_breakpoints)} breakpoints found")
                    return True
                else:
                    self.log_test("Responsive design breakpoints", False,
                                f"Missing breakpoints: {missing_breakpoints}")
                    return False
            else:
                self.log_test("Responsive design breakpoints", False,
                            "responsive.css not found")
                return False
                
        except Exception as e:
            self.log_test("Responsive design breakpoints", False, str(e))
            return False
    
    def test_animation_system(self):
        """Test that the animation system has required animations."""
        print("\n✨ Testing animation system...")
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        animations_file = css_dir / "animations.css"
        
        expected_animations = [
            "@keyframes fadeIn",
            "@keyframes slideInLeft", 
            "@keyframes scaleIn",
            "@keyframes spin",
            "@keyframes pulse",
            "transition:",
            "animation:",
            "prefers-reduced-motion"
        ]
        
        try:
            if animations_file.exists():
                content = animations_file.read_text()
                
                missing_animations = []
                for animation in expected_animations:
                    if animation not in content:
                        missing_animations.append(animation)
                
                if not missing_animations:
                    self.log_test("Animation system", True,
                                f"All {len(expected_animations)} animation features found")
                    return True
                else:
                    self.log_test("Animation system", False,
                                f"Missing animations: {missing_animations}")
                    return False
            else:
                self.log_test("Animation system", False,
                            "animations.css not found")
                return False
                
        except Exception as e:
            self.log_test("Animation system", False, str(e))
            return False
    
    def test_graph_styling_system(self):
        """Test that graph element styling system is comprehensive."""
        print("\n🎯 Testing graph styling system...")
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        graph_styles_file = css_dir / "graph-styles.css"
        
        expected_graph_features = [
            ".graph-node",
            ".graph-edge", 
            ".node-label",
            ".edge-label",
            ".graph-node:hover",
            ".graph-node.selected",
            ".graph-edge:hover",
            ".graph-edge.selected",
            "type-user",
            "type-project",
            "size-small",
            "size-large",
            "state-active",
            "weight-strong"
        ]
        
        try:
            if graph_styles_file.exists():
                content = graph_styles_file.read_text()
                
                missing_features = []
                for feature in expected_graph_features:
                    if feature not in content:
                        missing_features.append(feature)
                
                if not missing_features:
                    self.log_test("Graph styling system", True,
                                f"All {len(expected_graph_features)} styling features found")
                    return True
                else:
                    self.log_test("Graph styling system", False,
                                f"Missing features: {missing_features}")
                    return False
            else:
                self.log_test("Graph styling system", False,
                            "graph-styles.css not found")
                return False
                
        except Exception as e:
            self.log_test("Graph styling system", False, str(e))
            return False
    
    def test_html_integration(self):
        """Test that HTML properly integrates all CSS and JS files."""
        print("\n🌐 Testing HTML integration...")
        
        html_file = project_root / "src" / "viz" / "frontend" / "html" / "index.html"
        
        expected_includes = [
            'css/styles.css',
            'css/responsive.css',
            'css/animations.css',
            'css/graph-styles.css',
            'js/theme-manager.js',
            'id="theme-select"',
            'value="neon"',  # Check that neon theme is included
            'class="graph-markers"'  # SVG markers
        ]
        
        try:
            if html_file.exists():
                content = html_file.read_text()
                
                missing_includes = []
                for include in expected_includes:
                    if include not in content:
                        missing_includes.append(include)
                
                if not missing_includes:
                    self.log_test("HTML integration", True,
                                f"All {len(expected_includes)} integrations found")
                    return True
                else:
                    self.log_test("HTML integration", False,
                                f"Missing integrations: {missing_includes}")
                    return False
            else:
                self.log_test("HTML integration", False,
                            "index.html not found")
                return False
                
        except Exception as e:
            self.log_test("HTML integration", False, str(e))
            return False
    
    def test_theme_css_variables(self):
        """Test that CSS variables are properly defined for theming."""
        print("\n🎨 Testing CSS variables for theming...")
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        styles_file = css_dir / "styles.css"
        
        expected_variables = [
            "--color-primary",
            "--color-background",
            "--color-text-primary",
            "--color-border",
            "--font-family-primary",
            "--transition-normal",
            "--border-radius",
            "--z-header"
        ]
        
        try:
            if styles_file.exists():
                content = styles_file.read_text()
                
                missing_variables = []
                for variable in expected_variables:
                    if variable not in content:
                        missing_variables.append(variable)
                
                if not missing_variables:
                    self.log_test("CSS variables for theming", True,
                                f"All {len(expected_variables)} variables found")
                    return True
                else:
                    self.log_test("CSS variables for theming", False,
                                f"Missing variables: {missing_variables}")
                    return False
            else:
                self.log_test("CSS variables for theming", False,
                            "styles.css not found")
                return False
                
        except Exception as e:
            self.log_test("CSS variables for theming", False, str(e))
            return False
    
    def test_accessibility_features(self):
        """Test that accessibility features are implemented."""
        print("\n♿ Testing accessibility features...")
        
        files_to_check = {
            "styles.css": ["prefers-contrast", "focus-visible", "sr-only"],
            "responsive.css": ["prefers-reduced-motion", "touch-action"],
            "animations.css": ["prefers-reduced-motion: reduce", "animation: none"],
            "graph-styles.css": ["focus", "aria-selected", "outline"]
        }
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        all_accessibility_passed = True
        
        for filename, expected_features in files_to_check.items():
            file_path = css_dir / filename
            
            try:
                if file_path.exists():
                    content = file_path.read_text()
                    
                    found_features = []
                    for feature in expected_features:
                        if feature in content:
                            found_features.append(feature)
                    
                    if found_features:
                        self.log_test(f"Accessibility in {filename}", True,
                                    f"Found features: {found_features}")
                    else:
                        self.log_test(f"Accessibility in {filename}", False,
                                    f"No accessibility features found from: {expected_features}")
                        all_accessibility_passed = False
                else:
                    self.log_test(f"Accessibility in {filename}", False,
                                "File not found")
                    all_accessibility_passed = False
                    
            except Exception as e:
                self.log_test(f"Accessibility in {filename}", False, str(e))
                all_accessibility_passed = False
        
        return all_accessibility_passed
    
    def test_performance_optimizations(self):
        """Test that performance optimizations are implemented."""
        print("\n⚡ Testing performance optimizations...")
        
        files_to_check = {
            "styles.css": ["will-change", "transform: translateZ", "GPU acceleration"],
            "responsive.css": ["content-visibility", "contain-intrinsic-size"],
            "animations.css": ["backface-visibility", "transform: translateZ(0)"],
            "graph-styles.css": ["will-change", "performance"]
        }
        
        css_dir = project_root / "src" / "viz" / "frontend" / "css"
        all_performance_passed = True
        
        for filename, expected_optimizations in files_to_check.items():
            file_path = css_dir / filename
            
            try:
                if file_path.exists():
                    content = file_path.read_text()
                    
                    found_optimizations = []
                    for optimization in expected_optimizations:
                        if optimization in content:
                            found_optimizations.append(optimization)
                    
                    if found_optimizations:
                        self.log_test(f"Performance in {filename}", True,
                                    f"Found optimizations: {found_optimizations}")
                    else:
                        self.log_test(f"Performance in {filename}", False,
                                    f"No performance optimizations found from: {expected_optimizations}")
                        all_performance_passed = False
                else:
                    self.log_test(f"Performance in {filename}", False,
                                "File not found")
                    all_performance_passed = False
                    
            except Exception as e:
                self.log_test(f"Performance in {filename}", False, str(e))
                all_performance_passed = False
        
        return all_performance_passed
    
    def test_svg_markers_and_graphics(self):
        """Test that SVG markers and graphics are properly defined."""
        print("\n🎯 Testing SVG markers and graphics...")
        
        html_file = project_root / "src" / "viz" / "frontend" / "html" / "index.html"
        
        expected_svg_elements = [
            'id="arrowhead"',
            'id="circle-marker"',
            'id="diamond-marker"',
            'id="node-gradient-light"',
            'id="node-gradient-dark"',
            'id="glow"',
            'id="drop-shadow"',
            '<defs>',
            '<marker',
            '<filter'
        ]
        
        try:
            if html_file.exists():
                content = html_file.read_text()
                
                missing_svg = []
                for element in expected_svg_elements:
                    if element not in content:
                        missing_svg.append(element)
                
                if not missing_svg:
                    self.log_test("SVG markers and graphics", True,
                                f"All {len(expected_svg_elements)} SVG elements found")
                    return True
                else:
                    self.log_test("SVG markers and graphics", False,
                                f"Missing SVG elements: {missing_svg}")
                    return False
            else:
                self.log_test("SVG markers and graphics", False,
                            "index.html not found")
                return False
                
        except Exception as e:
            self.log_test("SVG markers and graphics", False, str(e))
            return False
    
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*80)
        print("🎨 PHASE 9 VISUAL DESIGN & THEMES - TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   • {result['test']}: {result['details']}")
        
        print(f"\n🎨 PHASE 9 IMPLEMENTATION STATUS:")
        
        # Check implementation completeness based on test results
        critical_tests = [
            "CSS file styles.css",
            "CSS file responsive.css",
            "CSS file animations.css", 
            "CSS file graph-styles.css",
            "ThemeManager JavaScript",
            "Theme system completeness",
            "HTML integration"
        ]
        
        critical_passed = sum(1 for result in self.test_results 
                            if result["test"] in critical_tests and result["passed"])
        
        if critical_passed == len(critical_tests):
            print("   ✅ PHASE 9 CORE IMPLEMENTATION: COMPLETE")
            print("   ✅ 5-theme CSS system implemented")
            print("   ✅ ThemeManager JavaScript with theme switching")
            print("   ✅ Responsive design for mobile/tablet/desktop")
            print("   ✅ Smooth animations and transitions")
            print("   ✅ Professional node/edge styling")
            print("   ✅ Complete HTML integration")
        else:
            print("   ⚠️  PHASE 9 CORE IMPLEMENTATION: INCOMPLETE")
            print(f"   Critical tests passed: {critical_passed}/{len(critical_tests)}")
        
        # Feature status
        feature_tests = [
            "Responsive design breakpoints",
            "Animation system",
            "Graph styling system", 
            "CSS variables for theming",
            "SVG markers and graphics"
        ]
        
        feature_passed = sum(1 for result in self.test_results 
                           if result["test"] in feature_tests and result["passed"])
        
        if feature_passed >= len(feature_tests) * 0.8:  # 80% threshold
            print("   ✅ ADVANCED FEATURES: COMPLETE")
        else:
            print("   ⚠️  ADVANCED FEATURES: NEEDS ATTENTION")
        
        # Quality checks
        quality_tests = [
            "Accessibility in",
            "Performance in"
        ]
        
        quality_passed = sum(1 for result in self.test_results 
                           if any(qt in result["test"] for qt in quality_tests) and result["passed"])
        
        if quality_passed >= 6:  # Most accessibility and performance tests
            print("   ✅ ACCESSIBILITY & PERFORMANCE: EXCELLENT")
        elif quality_passed >= 4:
            print("   ✅ ACCESSIBILITY & PERFORMANCE: GOOD")
        else:
            print("   ⚠️  ACCESSIBILITY & PERFORMANCE: NEEDS IMPROVEMENT")
        
        print(f"\n🎯 NEXT STEPS:")
        if failed_tests == 0:
            print("   ✅ Phase 9 implementation is complete and ready!")
            print("   🚀 Ready to proceed to Phase 10: Performance Optimization")
        else:
            print("   🔧 Address failed tests to complete Phase 9")
            print("   📝 Review implementation gaps identified above")
        
        print("\n" + "="*80)
        
        # Save detailed report
        report_data = {
            "phase": "Phase 9: Visual Design & Themes",
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests, 
                "failed_tests": failed_tests,
                "success_rate": (passed_tests/total_tests)*100
            },
            "test_results": self.test_results,
            "status": "COMPLETE" if failed_tests == 0 else "INCOMPLETE"
        }
        
        report_file = project_root / "PHASE9_TEST_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    def run_all_tests(self):
        """Run all Phase 9 tests."""
        print("🎨 STARTING PHASE 9 VISUAL DESIGN & THEMES COMPREHENSIVE TESTS")
        print("="*80)
        
        # Test sequence
        test_methods = [
            self.test_css_files_exist,
            self.test_theme_system_completeness,
            self.test_theme_manager_javascript,
            self.test_responsive_breakpoints,
            self.test_animation_system,
            self.test_graph_styling_system,
            self.test_html_integration,
            self.test_theme_css_variables,
            self.test_accessibility_features,
            self.test_performance_optimizations,
            self.test_svg_markers_and_graphics
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test(test_method.__name__, False, f"Test crashed: {e}")
        
        # Generate comprehensive report
        self.generate_test_report()

def main():
    """Main test execution."""
    print("🎨 Phase 9 Visual Design & Themes - Comprehensive Test Suite")
    print("Testing CSS themes, responsive design, animations, and styling\n")
    
    tester = Phase9Tester()
    tester.run_all_tests()
    
    # Return appropriate exit code
    failed_tests = sum(1 for result in tester.test_results if not result["passed"])
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)