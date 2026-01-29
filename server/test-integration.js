#!/usr/bin/env node
/**
 * Integration Test for Synthetic Data Generator
 * 
 * Tests the full stack:
 * - API server startup
 * - Job creation
 * - Progress tracking
 * - Job status retrieval
 */

const axios = require('axios');

const API_BASE = 'http://localhost:3001/api';
const TEST_TIMEOUT = 60000; // 1 minute

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testHealthCheck() {
  console.log('\n🔍 Testing health check...');
  try {
    const response = await axios.get(`${API_BASE}/health`);
    console.log('✅ Health check passed:', response.data);
    return true;
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
    return false;
  }
}

async function testTemplatesList() {
  console.log('\n🔍 Testing templates list...');
  try {
    const response = await axios.get(`${API_BASE}/templates`);
    console.log(`✅ Templates loaded: ${response.data.templates.length} templates`);
    response.data.templates.forEach(t => {
      console.log(`   - ${t.name} (${t.category})`);
    });
    return true;
  } catch (error) {
    console.error('❌ Templates list failed:', error.message);
    return false;
  }
}

async function testJobCreation() {
  console.log('\n🔍 Testing job creation...');
  try {
    const config = {
      domain: 'technology',
      targetCount: 50,  // Small test job
      batchSize: 10,
      outputFormat: 'jsonl',
      domainDescription: 'Simple programming Q&A for testing',
      topics: ['Python basics', 'JavaScript fundamentals']
    };
    
    const response = await axios.post(`${API_BASE}/generate`, config);
    console.log('✅ Job created:', response.data);
    
    return response.data.jobId;
  } catch (error) {
    console.error('❌ Job creation failed:', error.response?.data || error.message);
    return null;
  }
}

async function testJobStatus(jobId) {
  console.log('\n🔍 Testing job status tracking...');
  let attempts = 0;
  const maxAttempts = 10;
  
  while (attempts < maxAttempts) {
    try {
      const response = await axios.get(`${API_BASE}/jobs/${jobId}`);
      const job = response.data;
      
      console.log(`   Status: ${job.status} | Progress: ${job.progress?.toFixed(1) || 0}% | Generated: ${job.generated}/${job.targetCount}`);
      
      if (job.status === 'completed') {
        console.log('✅ Job completed successfully!');
        return true;
      } else if (job.status === 'failed') {
        console.error('❌ Job failed:', job.error);
        return false;
      }
      
      attempts++;
      await sleep(3000); // Check every 3 seconds
    } catch (error) {
      console.error('❌ Job status check failed:', error.message);
      return false;
    }
  }
  
  console.log('⚠️ Job still running after timeout');
  return false;
}

async function testJobsList() {
  console.log('\n🔍 Testing jobs list...');
  try {
    const response = await axios.get(`${API_BASE}/jobs`);
    console.log(`✅ Jobs list retrieved: ${response.data.jobs.length} jobs`);
    return true;
  } catch (error) {
    console.error('❌ Jobs list failed:', error.message);
    return false;
  }
}

async function testDomainCreation() {
  console.log('\n🔍 Testing custom domain creation...');
  try {
    const domain = {
      name: 'Test Custom Domain',
      description: 'A test domain for integration testing',
      topics: ['Topic 1', 'Topic 2', 'Topic 3'],
      questionTypes: ['definition', 'explanation'],
      difficultyLevels: ['beginner', 'intermediate']
    };
    
    const response = await axios.post(`${API_BASE}/domains`, domain);
    console.log('✅ Domain created:', response.data);
    return response.data.id;
  } catch (error) {
    console.error('❌ Domain creation failed:', error.response?.data || error.message);
    return null;
  }
}

async function runTests() {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║  Synthetic Data Generator Integration Tests   ║');
  console.log('╚════════════════════════════════════════════════╝');
  
  const results = {
    passed: 0,
    failed: 0,
    total: 0
  };
  
  // Test 1: Health Check
  results.total++;
  if (await testHealthCheck()) results.passed++; else results.failed++;
  
  // Test 2: Templates List
  results.total++;
  if (await testTemplatesList()) results.passed++; else results.failed++;
  
  // Test 3: Domain Creation
  results.total++;
  const domainId = await testDomainCreation();
  if (domainId) results.passed++; else results.failed++;
  
  // Test 4: Job Creation
  results.total++;
  const jobId = await testJobCreation();
  if (jobId) {
    results.passed++;
    
    // Test 5: Job Status Tracking
    // Note: This test will likely timeout or fail if Python/GPU not available
    // That's OK - we're just testing the API layer
    results.total++;
    if (await testJobStatus(jobId)) results.passed++; else results.failed++;
  } else {
    results.failed++;
  }
  
  // Test 6: Jobs List
  results.total++;
  if (await testJobsList()) results.passed++; else results.failed++;
  
  // Summary
  console.log('\n');
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║              Test Results                      ║');
  console.log('╚════════════════════════════════════════════════╝');
  console.log(`✅ Passed: ${results.passed}/${results.total}`);
  console.log(`❌ Failed: ${results.failed}/${results.total}`);
  console.log('');
  
  if (results.failed > 0) {
    console.log('⚠️  Some tests failed. Make sure the server is running on port 3001');
    console.log('   Start it with: cd server && npm start');
    process.exit(1);
  } else {
    console.log('🎉 All tests passed!');
    process.exit(0);
  }
}

// Check if server is running
async function checkServerRunning() {
  try {
    await axios.get(`${API_BASE}/health`, { timeout: 2000 });
    return true;
  } catch {
    return false;
  }
}

// Main
(async () => {
  const isRunning = await checkServerRunning();
  if (!isRunning) {
    console.error('❌ Server is not running on port 3001');
    console.error('   Start it with: cd server && npm start');
    process.exit(1);
  }
  
  await runTests();
})();
