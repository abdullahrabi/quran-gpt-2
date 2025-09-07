#!/usr/bin/env python3
"""
Test script to verify the setup and configuration.
Run this before running the main ingestion script.
"""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test if environment variables are properly set"""
    print("🔍 Testing environment variables...")
    
    load_dotenv()
    
    # Check Google API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        print("✅ GOOGLE_API_KEY found")
        if len(google_key) > 20:  # Basic validation
            print("✅ Google API key appears valid")
        else:
            print("⚠️  Google API key seems too short")
    else:
        print("❌ GOOGLE_API_KEY not found")
        return False
    
    # Check Pinecone API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        print("✅ PINECONE_API_KEY found")
        if len(pinecone_key) > 20:  # Basic validation
            print("✅ Pinecone API key appears valid")
        else:
            print("⚠️  Pinecone API key seems too short")
    else:
        print("❌ PINECONE_API_KEY not found")
        return False
    
    return True

def test_dependencies():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        "langchain",
        "langchain_google_genai", 
        "google.generativeai",
        "pinecone",
        "dotenv",
        "PyPDF2"
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            all_good = False
    
    return all_good

def test_data_folder():
    """Test if the Data folder exists and contains files"""
    print("\n🔍 Testing Data folder...")
    
    data_path = "Data"
    if os.path.exists(data_path):
        print(f"✅ Data folder found: {data_path}")
        
        files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        if files:
            print(f"✅ Found {len(files)} file(s) in Data folder:")
            for file in files:
                print(f"   - {file}")
        else:
            print("⚠️  Data folder is empty")
            return False
    else:
        print(f"❌ Data folder not found: {data_path}")
        return False
    
    return True

def test_google_connection():
    """Test Google API connection"""
    print("\n🔍 Testing Google API connection...")
    
    try:
        import google.generativeai as genai
        
        # Set up the API key
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Test with a simple model list request
        models = genai.list_models()
        print("✅ Google API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Google API connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running setup tests...\n")
    
    tests = [
        ("Environment Variables", test_environment),
        ("Dependencies", test_dependencies),
        ("Data Folder", test_data_folder),
        ("Google API Connection", test_google_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the ingestion script.")
        print("Run: python ingest.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues before running the ingestion script.")
        return 1

if __name__ == "__main__":
    exit(main())
