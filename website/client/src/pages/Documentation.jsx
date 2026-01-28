import { useState } from 'react';
import { 
  Book, Code, Zap, Terminal, FileText, 
  ChevronRight, Copy, Check, ExternalLink
} from 'lucide-react';

const Documentation = () => {
  const [activeSection, setActiveSection] = useState('getting-started');
  const [copiedCode, setCopiedCode] = useState(null);

  const sections = [
    { id: 'getting-started', title: 'Getting Started', icon: <Zap className="w-4 h-4" /> },
    { id: 'quick-start', title: 'Quick Start', icon: <Terminal className="w-4 h-4" /> },
    { id: 'api-reference', title: 'API Reference', icon: <Code className="w-4 h-4" /> },
    { id: 'configuration', title: 'Configuration', icon: <FileText className="w-4 h-4" /> },
    { id: 'output-format', title: 'Output Format', icon: <Book className="w-4 h-4" /> },
  ];

  const copyToClipboard = (code, id) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const CodeBlock = ({ code, language = 'bash', id }) => (
    <div className="relative group">
      <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
        <code className={`language-${language} text-sm text-gray-300`}>{code}</code>
      </pre>
      <button
        onClick={() => copyToClipboard(code, id)}
        className="absolute top-3 right-3 p-2 bg-slate-700 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copiedCode === id ? (
          <Check className="w-4 h-4 text-green-400" />
        ) : (
          <Copy className="w-4 h-4 text-gray-400" />
        )}
      </button>
    </div>
  );

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar */}
          <aside className="lg:w-64 flex-shrink-0">
            <div className="sticky top-24">
              <h2 className="text-lg font-semibold mb-4">Documentation</h2>
              <nav className="space-y-1">
                {sections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center space-x-3 px-4 py-2 rounded-lg text-left transition-colors ${
                      activeSection === section.id
                        ? 'bg-purple-500/20 text-purple-300'
                        : 'text-gray-400 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    {section.icon}
                    <span>{section.title}</span>
                  </button>
                ))}
              </nav>

              <div className="mt-8 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                <h3 className="font-medium mb-2">Need Help?</h3>
                <p className="text-sm text-gray-400 mb-3">
                  Check our GitHub for issues and discussions.
                </p>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-sm text-purple-400 hover:text-purple-300"
                >
                  <span>View on GitHub</span>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 min-w-0">
            {activeSection === 'getting-started' && (
              <div className="prose prose-invert max-w-none">
                <h1 className="text-3xl font-bold mb-4">Getting Started</h1>
                <p className="text-gray-400 text-lg mb-8">
                  Welcome to Synthetic Data Generator! This guide will help you get up and running 
                  with generating high-quality synthetic datasets.
                </p>

                <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 border border-purple-500/20 rounded-xl p-6 mb-8">
                  <h3 className="text-xl font-semibold mb-2">What You'll Build</h3>
                  <p className="text-gray-300">
                    By the end of this guide, you'll be able to generate thousands of Q&A pairs 
                    for machine learning training, using either Google Colab or your local machine.
                  </p>
                </div>

                <h2 className="text-2xl font-semibold mt-8 mb-4">Prerequisites</h2>
                <ul className="list-disc list-inside space-y-2 text-gray-300">
                  <li>A Google Account (for Google Colab) - Free!</li>
                  <li>Basic familiarity with Python/Jupyter Notebooks</li>
                  <li><strong>Optional:</strong> Local GPU (RTX 3090/4090 or better recommended)</li>
                </ul>

                <h2 className="text-2xl font-semibold mt-8 mb-4">System Requirements</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-3 px-4">Hardware</th>
                        <th className="text-left py-3 px-4">Speed</th>
                        <th className="text-left py-3 px-4">10k Dataset</th>
                        <th className="text-left py-3 px-4">30k Dataset</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4">T4 GPU (Colab Free)</td>
                        <td className="py-3 px-4">~100/min</td>
                        <td className="py-3 px-4">~1.7 hours</td>
                        <td className="py-3 px-4">~5 hours</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4">A100 GPU (Colab Pro)</td>
                        <td className="py-3 px-4">~200/min</td>
                        <td className="py-3 px-4">~50 min</td>
                        <td className="py-3 px-4">~2.5 hours</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4">RTX 3090/4090</td>
                        <td className="py-3 px-4">~150/min</td>
                        <td className="py-3 px-4">~1.1 hours</td>
                        <td className="py-3 px-4">~3.5 hours</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4">CPU Only</td>
                        <td className="py-3 px-4">~10/min</td>
                        <td className="py-3 px-4">~16 hours</td>
                        <td className="py-3 px-4">~50 hours</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeSection === 'quick-start' && (
              <div className="prose prose-invert max-w-none">
                <h1 className="text-3xl font-bold mb-4">Quick Start</h1>
                <p className="text-gray-400 text-lg mb-8">
                  Get started in under 5 minutes with Google Colab.
                </p>

                <h2 className="text-2xl font-semibold mt-8 mb-4">Option 1: Google Colab (Recommended)</h2>
                <p className="text-gray-300 mb-4">
                  The easiest way to get started. No setup required - just run in the cloud!
                </p>
                <CodeBlock
                  id="colab"
                  language="bash"
                  code={`# 1. Upload the generator script to Colab
# 2. Configure runtime: Runtime > Change runtime type > T4 GPU
# 3. Run the generator:

!python financial_education_generator_ultra.py`}
                />
                <p className="text-sm text-gray-400 mt-2 mb-8">
                  ⏱️ First run: ~5 minutes to install dependencies and download model (~5GB)
                </p>

                <h2 className="text-2xl font-semibold mt-8 mb-4">Option 2: Local Machine</h2>
                <p className="text-gray-300 mb-4">
                  For users with local GPU setup.
                </p>
                <CodeBlock
                  id="local"
                  language="bash"
                  code={`# Clone the repository
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator

# Install dependencies
pip install transformers accelerate bitsandbytes torch tqdm

# Run the generator
python Pre-Work/financial_education_generator_ultra.py`}
                />

                <h2 className="text-2xl font-semibold mt-8 mb-4">Option 3: Universal Generator</h2>
                <p className="text-gray-300 mb-4">
                  For generating datasets in any domain, not just finance.
                </p>
                <CodeBlock
                  id="universal"
                  language="bash"
                  code={`python Pre-Work/universal_dataset_generator.py`}
                />
              </div>
            )}

            {activeSection === 'api-reference' && (
              <div className="prose prose-invert max-w-none">
                <h1 className="text-3xl font-bold mb-4">API Reference</h1>
                <p className="text-gray-400 text-lg mb-8">
                  RESTful API for programmatic dataset generation.
                </p>

                <h2 className="text-2xl font-semibold mt-8 mb-4">Base URL</h2>
                <CodeBlock
                  id="base-url"
                  language="text"
                  code={`https://api.synthgen.ai/v1`}
                />

                <h2 className="text-2xl font-semibold mt-8 mb-4">Endpoints</h2>

                <div className="space-y-6">
                  <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                    <div className="flex items-center space-x-2 mb-4">
                      <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-sm font-mono">POST</span>
                      <code className="text-purple-300">/generate</code>
                    </div>
                    <p className="text-gray-400 mb-4">Start a new dataset generation job.</p>
                    <h4 className="font-medium mb-2">Request Body</h4>
                    <CodeBlock
                      id="generate-req"
                      language="json"
                      code={`{
  "domain": "financial",
  "targetCount": 1000,
  "batchSize": 25,
  "outputFormat": "jsonl"
}`}
                    />
                    <h4 className="font-medium mb-2 mt-4">Response</h4>
                    <CodeBlock
                      id="generate-res"
                      language="json"
                      code={`{
  "jobId": "gen_abc123",
  "status": "running",
  "estimatedTime": "10 minutes"
}`}
                    />
                  </div>

                  <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                    <div className="flex items-center space-x-2 mb-4">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-sm font-mono">GET</span>
                      <code className="text-purple-300">/templates</code>
                    </div>
                    <p className="text-gray-400 mb-4">List all available domain templates.</p>
                    <h4 className="font-medium mb-2">Response</h4>
                    <CodeBlock
                      id="templates-res"
                      language="json"
                      code={`{
  "templates": [
    {
      "id": "fin-education",
      "name": "Financial Education",
      "category": "financial"
    }
  ]
}`}
                    />
                  </div>

                  <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                    <div className="flex items-center space-x-2 mb-4">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-sm font-mono">GET</span>
                      <code className="text-purple-300">/jobs/:jobId</code>
                    </div>
                    <p className="text-gray-400 mb-4">Get status of a generation job.</p>
                    <h4 className="font-medium mb-2">Response</h4>
                    <CodeBlock
                      id="job-res"
                      language="json"
                      code={`{
  "jobId": "gen_abc123",
  "status": "completed",
  "generated": 1000,
  "downloadUrl": "/downloads/gen_abc123.jsonl"
}`}
                    />
                  </div>
                </div>
              </div>
            )}

            {activeSection === 'configuration' && (
              <div className="prose prose-invert max-w-none">
                <h1 className="text-3xl font-bold mb-4">Configuration</h1>
                <p className="text-gray-400 text-lg mb-8">
                  Customize the generator by editing the configuration options.
                </p>

                <h2 className="text-2xl font-semibold mt-8 mb-4">ExtremeSpeedConfig</h2>
                <p className="text-gray-300 mb-4">
                  The main configuration class for controlling generation behavior.
                </p>
                <CodeBlock
                  id="config"
                  language="python"
                  code={`@dataclass
class ExtremeSpeedConfig:
    # Model Settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_quantization: bool = True      # 4-bit for T4 GPU
    use_flash_attention: bool = True   # FlashAttention 2

    # Generation Settings
    batch_size: int = 25               # Q&A pairs per LLM call
    target_count: int = 30000          # Total pairs to generate
    save_interval: int = 200           # Buffer flush interval

    # Quality Settings
    min_answer_length: int = 40        # Minimum answer chars
    output_file: str = "dataset.jsonl" # Output filename`}
                />

                <h2 className="text-2xl font-semibold mt-8 mb-4">Configuration Options</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-3 px-4">Option</th>
                        <th className="text-left py-3 px-4">Type</th>
                        <th className="text-left py-3 px-4">Default</th>
                        <th className="text-left py-3 px-4">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">model_name</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">Mistral-7B</td>
                        <td className="py-3 px-4">HuggingFace model identifier</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">batch_size</td>
                        <td className="py-3 px-4">int</td>
                        <td className="py-3 px-4">25</td>
                        <td className="py-3 px-4">Q&A pairs per LLM call</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">target_count</td>
                        <td className="py-3 px-4">int</td>
                        <td className="py-3 px-4">30000</td>
                        <td className="py-3 px-4">Total pairs to generate</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">use_quantization</td>
                        <td className="py-3 px-4">bool</td>
                        <td className="py-3 px-4">true</td>
                        <td className="py-3 px-4">Enable 4-bit quantization</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-mono text-purple-300">min_answer_length</td>
                        <td className="py-3 px-4">int</td>
                        <td className="py-3 px-4">40</td>
                        <td className="py-3 px-4">Minimum answer characters</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeSection === 'output-format' && (
              <div className="prose prose-invert max-w-none">
                <h1 className="text-3xl font-bold mb-4">Output Format</h1>
                <p className="text-gray-400 text-lg mb-8">
                  Generated datasets are saved in ML-ready JSONL format.
                </p>

                <h2 className="text-2xl font-semibold mt-8 mb-4">JSONL Structure</h2>
                <p className="text-gray-300 mb-4">
                  Each line in the output file is a valid JSON object:
                </p>
                <CodeBlock
                  id="output"
                  language="json"
                  code={`{
  "id": "fin_Per_Bud_1234_56789",
  "topic": "Personal Finance",
  "subtopic": "Budgeting Basics",
  "question": "What is the 50/30/20 rule in budgeting?",
  "answer": "The 50/30/20 rule is a simple budgeting framework that suggests allocating 50% of after-tax income to needs, 30% to wants, and 20% to savings and debt repayment...",
  "difficulty": "beginner",
  "question_type": "definition",
  "created_at": "2026-01-28T10:30:00.000000"
}`}
                />

                <h2 className="text-2xl font-semibold mt-8 mb-4">Field Descriptions</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-3 px-4">Field</th>
                        <th className="text-left py-3 px-4">Type</th>
                        <th className="text-left py-3 px-4">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">id</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">Unique identifier for the Q&A pair</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">topic</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">Main topic category</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">subtopic</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">Specific subtopic within the main topic</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">question</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">The generated question</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">answer</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">The generated answer</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">difficulty</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">beginner | intermediate | advanced</td>
                      </tr>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-3 px-4 font-mono text-purple-300">question_type</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">Type of question (definition, explanation, etc.)</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-mono text-purple-300">created_at</td>
                        <td className="py-3 px-4">string</td>
                        <td className="py-3 px-4">ISO 8601 timestamp</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <h2 className="text-2xl font-semibold mt-8 mb-4">Export Options</h2>
                <p className="text-gray-300 mb-4">
                  The generated JSONL can be easily converted to other formats:
                </p>
                <CodeBlock
                  id="convert"
                  language="python"
                  code={`import pandas as pd
import json

# Read JSONL
data = [json.loads(line) for line in open('dataset.jsonl')]
df = pd.DataFrame(data)

# Export to CSV
df.to_csv('dataset.csv', index=False)

# Export to Parquet (recommended for large datasets)
df.to_parquet('dataset.parquet')`}
                />
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
