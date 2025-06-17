import sys
import os
from lxml import etree
import argparse
from ftfy import fix_text
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QLabel,
    QCheckBox,
)


def strip_xml_tags(xml_file):
    """Removes XML tags from the file and returns plain text content."""
    try:
        with open(xml_file, "r", encoding="utf-8") as f:
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(f, parser)
            text = tree.xpath("string()")  # Extracts all text content
            return text.strip()
    except Exception as e:
        return f"Error processing {xml_file}: {str(e)}"


def process_xml_file(xml_file, repair=False, no_blank_lines=False):
    """Processes a single XML file, applies options, and writes the extracted text to a .txt file."""
    text = strip_xml_tags(xml_file)
    if text.startswith("Error"):
        print(text)
        return

    if repair:
        text = fix_text(text)  # Fixes encoding issues

    if no_blank_lines:
        text = "\n".join(line for line in text.splitlines() if line.strip())

    output_file = os.path.splitext(xml_file)[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Processed: {xml_file} -> {output_file}")


def cli_mode():
    """Handles the command-line interface mode."""
    parser = argparse.ArgumentParser(
        description="Convert XML to text while removing tags."
    )
    parser.add_argument("files", nargs="+", help="List of XML files to process.")
    parser.add_argument(
        "--repair", action="store_true", help="Fix text encoding issues using ftfy."
    )
    parser.add_argument(
        "--no-blank-lines",
        action="store_true",
        help="Remove blank lines before saving.",
    )

    args = parser.parse_args()

    for file in args.files:
        if file.endswith(".xml") and os.path.isfile(file):
            process_xml_file(
                file, repair=args.repair, no_blank_lines=args.no_blank_lines
            )
        else:
            print(f"Skipping: {file} (not a valid .xml file)")


class XMLTextExtractor(QWidget):
    """PySide6 GUI for XML to Text conversion."""

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("XML to Text Converter")
        self.setGeometry(100, 100, 500, 350)

        layout = QVBoxLayout()

        self.label = QLabel("Select XML files to process:")
        layout.addWidget(self.label)

        self.selectButton = QPushButton("Select XML Files")
        self.selectButton.clicked.connect(self.openFileDialog)
        layout.addWidget(self.selectButton)

        self.repairCheckbox = QCheckBox("Repair text (ftfy)")
        layout.addWidget(self.repairCheckbox)

        self.noBlankCheckbox = QCheckBox("No blank lines")
        layout.addWidget(self.noBlankCheckbox)

        self.processButton = QPushButton("Convert to Text")
        self.processButton.clicked.connect(self.processFiles)
        self.processButton.setEnabled(False)
        layout.addWidget(self.processButton)

        self.textOutput = QTextEdit()
        self.textOutput.setReadOnly(True)
        layout.addWidget(self.textOutput)

        self.setLayout(layout)
        self.selected_files = []

    def openFileDialog(self):
        """Opens a file dialog for selecting XML files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select XML Files", "", "XML Files (*.xml)"
        )
        if files:
            self.selected_files = files
            self.processButton.setEnabled(True)
            self.textOutput.append(f"Selected files:\n" + "\n".join(files))

    def processFiles(self):
        """Processes selected XML files."""
        if not self.selected_files:
            self.textOutput.append("No files selected.")
            return

        repair = self.repairCheckbox.isChecked()
        no_blank_lines = self.noBlankCheckbox.isChecked()

        self.textOutput.append("\nProcessing files...\n")
        for file in self.selected_files:
            process_xml_file(file, repair=repair, no_blank_lines=no_blank_lines)
            self.textOutput.append(f"Processed: {file}")

        self.textOutput.append("\nProcessing complete!")


def main():
    """Determines whether to run in GUI or CLI mode."""
    if len(sys.argv) > 1:
        cli_mode()
    else:
        app = QApplication(sys.argv)
        window = XMLTextExtractor()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
