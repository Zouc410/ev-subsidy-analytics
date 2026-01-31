"""
Presentation Slide Generator
Creates publication-ready slides for MIT MBAn interview
Combines visualizations into professional presentation format
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np

# Professional color scheme
COLORS = {
    'mit_red': '#A31F34',
    'dark_gray': '#1F2937',
    'light_gray': '#F3F4F6',
    'white': '#FFFFFF',
    'accent': '#2E86AB',
}


class SlideGenerator:
    """Generates professional presentation slides."""

    def __init__(self, output_dir: str = "output/slides"):
        """Initialize slide generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = Path("output/charts")

    def create_slide_1(self) -> str:
        """
        Create Slide 1: Project Overview & Business Impact
        Shows architecture + key metrics
        """
        fig = plt.figure(figsize=(16, 9), facecolor=COLORS['white'])

        # Title bar
        title_ax = fig.add_axes([0, 0.88, 1, 0.12])
        title_ax.set_facecolor(COLORS['mit_red'])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, 'EV Subsidy Analytics Pipeline',
                     transform=title_ax.transAxes, ha='center', va='center',
                     fontsize=28, fontweight='bold', color=COLORS['white'])
        title_ax.text(0.5, 0.15, 'Scalable Data Infrastructure for Policy Intelligence',
                     transform=title_ax.transAxes, ha='center', va='center',
                     fontsize=14, color=COLORS['white'], alpha=0.9)

        # Main content area
        gs = GridSpec(2, 3, figure=fig, left=0.05, right=0.95,
                     top=0.85, bottom=0.1, wspace=0.2, hspace=0.25)

        # Left: Architecture diagram
        ax_arch = fig.add_subplot(gs[:, 0])
        arch_file = self.charts_dir / 'architecture.png'
        if arch_file.exists():
            img = mpimg.imread(str(arch_file))
            ax_arch.imshow(img)
        ax_arch.axis('off')
        ax_arch.set_title('Pipeline Architecture', fontsize=14, fontweight='bold', pad=10)

        # Top right: Key Metrics
        ax_metrics = fig.add_subplot(gs[0, 1:])
        ax_metrics.axis('off')
        ax_metrics.set_title('Business Impact Metrics', fontsize=14, fontweight='bold', pad=10)

        # Metric cards
        metrics = [
            ('80,000x', 'Processing\nSpeedup', COLORS['mit_red']),
            ('98%', 'Data Quality\nScore', COLORS['accent']),
            ('18', 'States\nAnalyzed', COLORS['dark_gray']),
            ('$141M', 'Estimated\nAnnual Impact', '#28A745'),
        ]

        for i, (value, label, color) in enumerate(metrics):
            x = 0.12 + i * 0.22
            # Value
            ax_metrics.text(x, 0.65, value, ha='center', va='center',
                          fontsize=32, fontweight='bold', color=color)
            # Label
            ax_metrics.text(x, 0.25, label, ha='center', va='center',
                          fontsize=11, color=COLORS['dark_gray'])

        # Bottom right: Regional comparison
        ax_regional = fig.add_subplot(gs[1, 1:])
        regional_file = self.charts_dir / 'regional_comparison.png'
        if regional_file.exists():
            img = mpimg.imread(str(regional_file))
            ax_regional.imshow(img)
        ax_regional.axis('off')

        # Footer
        footer_ax = fig.add_axes([0, 0, 1, 0.05])
        footer_ax.set_facecolor(COLORS['light_gray'])
        footer_ax.axis('off')
        footer_ax.text(0.02, 0.5, 'Charlotte | MIT MBAn Interview',
                      transform=footer_ax.transAxes, va='center',
                      fontsize=10, color=COLORS['dark_gray'])
        footer_ax.text(0.98, 0.5, 'Slide 1 of 2',
                      transform=footer_ax.transAxes, va='center', ha='right',
                      fontsize=10, color=COLORS['dark_gray'])

        filepath = self.output_dir / 'slide_1_overview.png'
        plt.savefig(filepath, dpi=150, facecolor=COLORS['white'],
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return str(filepath)

    def create_slide_2(self) -> str:
        """
        Create Slide 2: Technical Deep Dive & Results
        Shows data visualizations + pipeline metrics
        """
        fig = plt.figure(figsize=(16, 9), facecolor=COLORS['white'])

        # Title bar
        title_ax = fig.add_axes([0, 0.88, 1, 0.12])
        title_ax.set_facecolor(COLORS['accent'])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, 'Data Analytics & Technical Results',
                     transform=title_ax.transAxes, ha='center', va='center',
                     fontsize=28, fontweight='bold', color=COLORS['white'])
        title_ax.text(0.5, 0.15, 'From Raw Policy Documents to Real-Time Insights',
                     transform=title_ax.transAxes, ha='center', va='center',
                     fontsize=14, color=COLORS['white'], alpha=0.9)

        # Main content
        gs = GridSpec(2, 2, figure=fig, left=0.05, right=0.95,
                     top=0.85, bottom=0.1, wspace=0.15, hspace=0.2)

        # Top left: State rebates chart
        ax1 = fig.add_subplot(gs[0, 0])
        rebates_file = self.charts_dir / 'state_rebates.png'
        if rebates_file.exists():
            img = mpimg.imread(str(rebates_file))
            ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('State-Level Incentive Comparison', fontsize=12, fontweight='bold', pad=5)

        # Top right: Pipeline metrics
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_file = self.charts_dir / 'pipeline_metrics.png'
        if metrics_file.exists():
            img = mpimg.imread(str(metrics_file))
            ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('Pipeline Performance', fontsize=12, fontweight='bold', pad=5)

        # Bottom left: Time series
        ax3 = fig.add_subplot(gs[1, 0])
        ts_file = self.charts_dir / 'time_series.png'
        if ts_file.exists():
            img = mpimg.imread(str(ts_file))
            ax3.imshow(img)
        ax3.axis('off')
        ax3.set_title('Policy Evolution Timeline', fontsize=12, fontweight='bold', pad=5)

        # Bottom right: Key takeaways
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        ax4.set_title('Key Technical Achievements', fontsize=14, fontweight='bold', pad=10)

        takeaways = [
            '✓ Automated ingestion of multi-format policy data',
            '✓ Real-time normalization with 98% accuracy',
            '✓ AI-powered policy summarization',
            '✓ RESTful API with sub-50ms response time',
            '✓ Scalable architecture for 50+ states',
            '✓ Publication-ready analytics & visualizations',
        ]

        for i, text in enumerate(takeaways):
            y = 0.85 - i * 0.13
            ax4.text(0.05, y, text, transform=ax4.transAxes,
                    fontsize=13, va='center', color=COLORS['dark_gray'])

        # Tech stack badges
        ax4.text(0.05, 0.1, 'Tech Stack:', transform=ax4.transAxes,
                fontsize=11, fontweight='bold', color=COLORS['dark_gray'])

        badges = ['Python', 'Pandas', 'FastAPI', 'Transformers', 'Matplotlib']
        for i, badge in enumerate(badges):
            x = 0.25 + i * 0.14
            ax4.add_patch(mpatches.FancyBboxPatch(
                (x - 0.02, 0.04), 0.12, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=COLORS['light_gray'],
                edgecolor=COLORS['accent'],
                transform=ax4.transAxes,
                linewidth=1
            ))
            ax4.text(x + 0.04, 0.08, badge, transform=ax4.transAxes,
                    fontsize=9, ha='center', va='center', color=COLORS['dark_gray'])

        # Footer
        footer_ax = fig.add_axes([0, 0, 1, 0.05])
        footer_ax.set_facecolor(COLORS['light_gray'])
        footer_ax.axis('off')
        footer_ax.text(0.02, 0.5, 'Charlotte | MIT MBAn Interview',
                      transform=footer_ax.transAxes, va='center',
                      fontsize=10, color=COLORS['dark_gray'])
        footer_ax.text(0.98, 0.5, 'Slide 2 of 2',
                      transform=footer_ax.transAxes, va='center', ha='right',
                      fontsize=10, color=COLORS['dark_gray'])

        filepath = self.output_dir / 'slide_2_technical.png'
        plt.savefig(filepath, dpi=150, facecolor=COLORS['white'],
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return str(filepath)

    def create_all_slides(self) -> Dict[str, str]:
        """Generate all presentation slides."""
        print("Generating presentation slides...")

        slides = {
            'slide_1': self.create_slide_1(),
            'slide_2': self.create_slide_2(),
        }

        print(f"Generated {len(slides)} slides in {self.output_dir}")
        return slides


def generate_presentation():
    """Main function to generate presentation slides."""
    generator = SlideGenerator()
    return generator.create_all_slides()


if __name__ == "__main__":
    slides = generate_presentation()
    print("\nGenerated slides:")
    for name, path in slides.items():
        print(f"  {name}: {path}")
